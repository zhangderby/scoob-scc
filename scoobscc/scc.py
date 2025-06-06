from .math_module import xp, xcipy, ensure_np_array
from scoobscc import utils
from scoobscc import scoob_interface as scoobi

import scoobpy

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize, CenteredNorm

def run(data,
        INDIclient,
        camsci_stream, 
        dm_stream, 
        control_matrix,
        calibration_modes,
        control_mask,
        scc_reference,
        shift,
        diam_window,
        im_params,
        ref_psf_params, 
        dark_frame, 
        NFRAMES=10, 
        delay=0.01,
        num_iterations=3,
        gain=0.75, 
        leakage=0.0,
    ):
    
    print('Running EFC...')

    Nmodes = calibration_modes.shape[0]
    Nmask = int(control_mask.sum())

    modes = calibration_modes.reshape(Nmodes, -1)

    command = copy.copy(data['commands'][-1])

    for i in range(num_iterations):
        print(f"\tIteration {i + 1} / {num_iterations}")

        estimate_vector = xp.zeros(2 * Nmask)

        estimate = take_measurement(I=I, scc_reference=scc_reference, shift=shift, diam_window=diam_window)
        estimate_vector[::2] = estimate[control_mask].ravel().real
        estimate_vector[1::2] = estimate[control_mask].ravel().imag

        del_modes = control_matrix.dot(estimate_vector)

        del_command = gain * del_modes.dot(modes).reshape(I.Nact, I.Nact)

        command = (1.0 - leakage) * command - del_command

        I.set_dm(command, channel=channel)

        image = I.snap_camsci()

        data['images'].append(copy.copy(image))
        data['commands'].append(copy.copy(command))

    return data


def sussy_calibrate(
        INDIclient,
        camsci_stream, 
        dm_stream, 
        control_mask, 
        calibration_amplitude, 
        calibration_modes,
        im_params,
        ref_psf_params, 
        scc_reference,
        shift,
        diam_window,
        dark_frame,
        NFRAMES=10,
        delay=0.01,
        scale_factors=None, 
        return_full_response=False,
    ):

    print('Sussy Calibrating Jacobian...')

    current_command = dm_stream.grab_latest() * 1e-6

    Nact = calibration_modes.shape[1]
    Nmask = int(control_mask.sum())
    Nmodes = calibration_modes.shape[0]
    Ncamsci = camsci_stream.shape[0]

    calib_amps = np.array([-calibration_amplitude, calibration_amplitude])

    ims_mod = np.zeros((Ncamsci, Ncamsci, Nmodes, 2))
    ims_unmod = np.zeros((Ncamsci, Ncamsci, Nmodes, 2))

    start = time.time()

    if INDIclient['stagepiezo.stagefold_pos.target'] == 0:
        print('Pinhole is blocked, moving stagepiezo...')
        scoobpy.utils.move_relative(client=INDIclient, device='stagepiezo.stagefold_pos', val=1000)
        time.sleep(5)

    for i, mode in enumerate(calibration_modes):

        for j, amp in enumerate(calib_amps):

            dm_mode = mode.reshape(Nact, Nact)
            amp = calibration_amplitude * scale_factors[i] if scale_factors is not None else calibration_amplitude
            dm_command = ensure_np_array(amp * dm_mode)

            dm_stream.write((current_command + dm_command) * 1e6)
            time.sleep(delay)

            im = scoobi.snap(camsci_stream, NFRAMES, dark_frame, im_params, ref_psf_params)
            im[im < 0] = 0
            ims_mod[:, :, i, j] = im

        print(f"\tSnapped modulated images of mode {i + 1:d}/{calibration_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")

    print('Finished taking modulated images, moving stagepiezo...')
    scoobpy.utils.move_relative(client=INDIclient, device='stagepiezo.stagefold_pos', val=-1000)
    time.sleep(5)

    for i, mode in enumerate(calibration_modes):

        for j, amp in enumerate(calib_amps):

            dm_mode = mode.reshape(Nact, Nact)
            amp = calibration_amplitude * scale_factors[i] if scale_factors is not None else calibration_amplitude
            dm_command = ensure_np_array(amp * dm_mode)

            dm_stream.write((current_command + dm_command) * 1e6)
            time.sleep(delay)

            im = scoobi.snap(camsci_stream, NFRAMES, dark_frame, im_params, ref_psf_params)
            im[im < 0] = 0
            ims_unmod[:, :, i, j] = im

        print(f"\tSnapped unmodulated images of mode {i + 1:d}/{calibration_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")

    response_matrix = xp.zeros((2 * Nmask, Nmodes))

    if return_full_response:
        response_matrix_full = xp.zeros((2 * Ncamsci ** 2, Nmodes))

    for i in range(Nmodes):

        response = 0

        for j, amp in enumerate(calib_amps):

            image_mod = xp.asarray(ims_mod[:, :, i, j])
            image_unmod = xp.asarray(ims_unmod[:, :, i, j])

            fft_mod = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(image_mod), norm='ortho'))
            fft_unmod = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(image_unmod), norm='ortho'))
            fft_diff = fft_mod - fft_unmod

            fft_shifted = xcipy.ndimage.shift(fft_diff, shift)

            x, y = xp.meshgrid(xp.linspace(-1, 1, fft_shifted.shape[0]), xp.linspace(-1, 1, fft_shifted.shape[0]))
            r = xp.sqrt(x ** 2 + y ** 2)
            mask = r < diam_window

            fft_masked = fft_shifted * mask

            estimate = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(fft_masked), norm='ortho'))
            estimate /= np.sqrt(scc_reference)

            response += amp * estimate / (2 * xp.var(calib_amps)) 

        response_matrix[::2, i] = response[control_mask].ravel().real
        response_matrix[1::2, i] = response[control_mask].ravel().imag
        
        if return_full_response:
            response_matrix_full[::2, i] = response.ravel().real
            response_matrix_full[1::2, i] = response.ravel().imag

        print(f"\tCalculated response of mode {i + 1:d}/{calibration_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")
    
    if return_full_response:
        return response_matrix, response_matrix_full
    else:
        return response_matrix



def compute_hadamard_scale_factors(had_modes, scale_exp=1/6, scale_thresh=4, iwa=2.5, owa=13, oversamp=4, plot=False):
    Nact = had_modes.shape[1]

    ft_modes = []
    for i in range(had_modes.shape[0]):
        had_mode = had_modes[i]
        ft_modes.append(xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(utils.pad_or_crop(had_mode, Nact*oversamp)))))
    mode_freqs = xp.abs(xp.array(ft_modes))

    mode_freq_mask_pxscl = 1/oversamp
    x = (xp.linspace(-Nact*oversamp//2, Nact*oversamp//2-1, Nact*oversamp) + 1/2)*mode_freq_mask_pxscl
    x,y = xp.meshgrid(x,x)
    r = xp.sqrt(x**2+y**2)
    mode_freq_mask = (r>iwa)*(r<owa)
    if plot: utils.imshow([mode_freq_mask], pxscls=[1/oversamp])

    sum_vals = []
    max_vals = []
    for i in range(had_modes.shape[0]):
        sum_vals.append(xp.sum(mode_freqs[i, mode_freq_mask]))
        max_vals.append(xp.max(mode_freqs[i, mode_freq_mask]**2))

    biggest_sum = xp.max(xp.array(sum_vals))
    biggest_max = xp.max(xp.array(max_vals))

    scale_factors = []
    for i in range(had_modes.shape[0]):
        scale_factors.append((biggest_max/max_vals[i])**scale_exp)
        # scale_factors.append((biggest_sum/sum_vals[i])**(1/2))
    scale_factors = ensure_np_array(xp.array(scale_factors))

    scale_factors[scale_factors>scale_thresh] = scale_thresh
    if plot: 
        plt.plot(scale_factors)
        plt.show()

    return scale_factors