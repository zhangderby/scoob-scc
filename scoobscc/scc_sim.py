from .math_module import xp, xcipy, ensure_np_array
from scoobscc import utils

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, CenteredNorm, Normalize, SymLogNorm
import matplotlib.pyplot as plt

# def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
def take_measurement(I, scc_reference, shift, diam_window, plot=False):

    I.LYOT = I.SCCSTOP

    image_mod = I.snap_camsci()

    I.LYOT = I.LYOTSTOP

    image_unmod = I.snap_camsci()

    fft_mod = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(image_mod), norm='ortho'))
    fft_unmod = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(image_unmod), norm='ortho'))
    fft_diff = fft_mod - fft_unmod

    if plot:
        plt.figure(figsize=(15, 4))
        plt.subplot(131)
        plt.imshow(xp.abs(fft_mod).get(), norm='log')
        plt.subplot(132)
        plt.imshow(xp.abs(fft_unmod).get(), norm='log')
        plt.subplot(133)
        plt.imshow(xp.abs(fft_diff).get(), norm='log')

    fft_shifted = xcipy.ndimage.shift(fft_diff, (0, shift))

    x, y = xp.meshgrid(xp.linspace(-1, 1, fft_shifted.shape[0]), xp.linspace(-1, 1, fft_shifted.shape[0]))
    r = xp.sqrt(x ** 2 + y ** 2)
    mask = r < diam_window

    fft_masked = fft_shifted * mask

    if plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.imshow(xp.abs(fft_shifted).get(), norm='log')
        plt.subplot(122)
        plt.imshow(xp.abs(fft_masked).get(), norm='log')

    estimate = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(fft_masked), norm='ortho'))
    estimate /= np.sqrt(scc_reference)

    if plot:
        plt.figure(figsize=(15, 4))
        plt.subplot(131)
        plt.imshow(xp.abs(estimate).get() ** 2, cmap='magma', norm='log', vmin=1e-9, vmax=1e-4)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Estimate')
        plt.subplot(132)
        plt.imshow(image_unmod.get(), cmap='magma', norm='log', vmin=1e-9, vmax=1e-4)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Truth')
        plt.subplot(133)
        plt.imshow(xp.abs(estimate).get() ** 2 - image_unmod.get(), cmap='magma', norm='log', vmin=1e-9, vmax=1e-4)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Difference')

    return estimate
    
def calibrate(
        I, 
        control_mask, 
        calibration_amplitude, 
        calibration_modes, 
        scc_reference,
        shift,
        diam_window,
        channel=3,
        return_full_response=False,
    ):

    print('Calibrating Jacobian...')

    Nmask = int(control_mask.sum())
    Nmodes = calibration_modes.shape[0]
    calib_amps = xp.array([-calibration_amplitude, calibration_amplitude])

    response_matrix = xp.zeros((2 * Nmask, Nmodes))

    if return_full_response:
        response_matrix_full = xp.zeros((2 * I.ncamsci ** 2, Nmodes))

    start = time.time()
    for i, mode in enumerate(calibration_modes):

        response = 0

        for amp in calib_amps:

            dm_command = mode.reshape(I.Nact, I.Nact)

            I.add_dm(amp * dm_command, channel=channel)

            estimate = take_measurement(I, scc_reference, shift, diam_window, plot=False)
            response += amp * estimate / (2 * xp.var(calib_amps))

            I.add_dm(-amp * dm_command, channel=channel)

        response_matrix[::2, i] = response[control_mask].ravel().real
        response_matrix[1::2, i] = response[control_mask].ravel().imag
        
        if return_full_response:
            response_matrix_full[::2, i] = response.ravel().real
            response_matrix_full[1::2, i] = response.ravel().imag

        print(f"\tCalibrated mode {i + 1:d}/{calibration_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")

    if return_full_response:
        return response_matrix, response_matrix_full
    else:
        return response_matrix
    
def run(I, 
        data,
        control_matrix,
        calibration_modes,
        control_mask,
        scc_reference,
        shift,
        diam_window,
        channel=3,
        num_iterations=3,
        gain=0.75, 
        leakage=0.0,
        plot=False,
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
    if plot: imshow1(mode_freq_mask, pxscl=1/oversamp)

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

import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize, CenteredNorm
from IPython.display import display, clear_output

def plot_data_with_ref(
        data, 
        im1vmin=1e-9, im1vmax=1e-4,
        im2vmin=1e-9, im2vmax=1e-4, 
        vmin=1e-9, vmax=1e-4, 
        xticks=None,
        exp_name='',
        fname=None,
    ):
    ims = ensure_np_array( xp.array(data['images']) ) 
    control_mask = ensure_np_array( data['control_mask'] )
    # print(type(control_mask))
    Nitr = ims.shape[0]
    npsf = ims.shape[1]
    psf_pixelscale_lamD = data['pixelscale']

    mean_nis = np.mean(ims[:,control_mask], axis=1)
    ibest = np.argmin(mean_nis)
    ref_im = ensure_np_array(data['images'][0])
    best_im = ensure_np_array(data['images'][ibest])

    fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(15,10), dpi=125, gridspec_kw={'width_ratios': [1, 1, 1], })
    ext = psf_pixelscale_lamD*npsf/2
    extent = [-ext, ext, -ext, ext]

    w = 0.225
    im1 = ax[0].imshow(ref_im, norm=LogNorm(vmax=im1vmax, vmin=im1vmin), cmap='magma', extent=extent)
    ax[0].set_title(f'Initial Image:\nMean Contrast = {mean_nis[0]:.2e}', fontsize=14)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im1, cax=cax)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)
    ax[0].set_position([0, 0, w, w]) # [left, bottom, width, height]

    im2 = ax[1].imshow( best_im, norm=LogNorm(vmax=im2vmax, vmin=im2vmin), cmap='magma', extent=extent)
    ax[1].set_title('Best Iteration' + exp_name + f':\nMean Contrast = {mean_nis[ibest]:.2e}', fontsize=14)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im2, cax=cax,)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)
    ax[1].set_position([0.23, 0, w, w])

    ax[0].set_ylabel('Y [$\lambda/D$]', fontsize=12, labelpad=-5)
    ax[0].set_xlabel('X [$\lambda/D$]', fontsize=12, labelpad=5)
    ax[1].set_xlabel('X [$\lambda/D$]', fontsize=12, labelpad=5)

    ax[2].set_title('Mean Contrast per Iteration' + exp_name, fontsize=14)
    ax[2].semilogy(mean_nis, label='3.6% Bandpass')
    ax[2].grid()
    ax[2].set_xlabel('Iteration Number', fontsize=12, )
    ax[2].set_ylabel('Mean Contrast', fontsize=14, labelpad=1)
    ax[2].set_ylim([vmin, vmax])
    xticks = np.arange(0,Nitr,2) if xticks is None else xticks
    ax[2].set_xticks(xticks)
    ax[2].set_position([0.525, 0, 0.3, w])

    if fname is not None: fig.savefig(fname, format='pdf', bbox_inches="tight")
