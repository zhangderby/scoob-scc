from .math_module import xp, xcipy, ensure_np_array
from scoobscc import utils
from scoobscc.imshows import imshow

import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
import poppy
import pickle

def make_mask(Nact=34):
    y,x = xp.indices((Nact, Nact)) - Nact//2 + 1/2
    r = xp.hypot(x, y)
    mask = r < (Nact/2 + 1/2)
    return mask

def make_gaussian_inf_fun(act_spacing=300e-6, sampling=10, coupling=0.15, Nact=4):
    ng = int(sampling*Nact)
    pxscl = act_spacing/(sampling)

    xs = (xp.linspace(-ng/2,ng/2-1,ng)+1/2) * pxscl
    x,y = xp.meshgrid(xs,xs)
    r = xp.sqrt(x**2 + y**2)

    d = act_spacing/np.sqrt(-np.log(coupling))

    inf_fun = np.exp(-(r/d)**2)

    return inf_fun

def create_hadamard_modes(dm_mask): 
    Nacts = dm_mask.sum().astype(int)
    Nact = dm_mask.shape[0]
    np2 = 2**int(xp.ceil(xp.log2(Nacts)))
    hmodes = xp.array(scipy.linalg.hadamard(np2))
    
    had_modes = []

    inds = xp.where(dm_mask.flatten().astype(int))
    for hmode in hmodes:
        hmode = hmode[:Nacts]
        mode = xp.zeros((dm_mask.shape[0]**2))
        mode[inds] = hmode
        had_modes.append(mode)
    had_modes = xp.array(had_modes).reshape(np2, Nact, Nact)
    
    return had_modes
    
def create_fourier_modes(
        dm_mask, 
        npsf, 
        psf_pixelscale_lamD, 
        iwa, 
        owa, 
        rotation=0, 
        fourier_sampling=0.75,
        which='both', 
        return_fs=False,
    ):
    Nact = dm_mask.shape[0]
    nfg = int(xp.round(npsf * psf_pixelscale_lamD/fourier_sampling))
    if nfg%2==1: nfg += 1
    yf, xf = (xp.indices((nfg, nfg)) - nfg//2 + 1/2) * fourier_sampling
    # fourier_cm = utils.create_annular_focal_plane_mask(nfg, fourier_sampling, iwa-fourier_sampling, owa+fourier_sampling, edge=iwa-fourier_sampling, rotation=rotation)
    fourier_cm = utils.create_annular_mask(
        nfg, 
        fourier_sampling, 
        iwa-fourier_sampling, 
        owa+fourier_sampling, 
        edge=iwa-fourier_sampling, 
        rotation=rotation,
    )
    ypp, xpp = (xp.indices((Nact, Nact)) - Nact//2 + 1/2)

    sampled_fs = xp.array([xf[fourier_cm], yf[fourier_cm]]).T

    fourier_modes = []
    for i in range(len(sampled_fs)):
        fx = sampled_fs[i,0]
        fy = sampled_fs[i,1]
        if which=='both' or which=='cos':
            fourier_modes.append( dm_mask * xp.cos(2 * np.pi * (fx*xpp + fy*ypp)/Nact) )
        if which=='both' or which=='sin':
            fourier_modes.append( dm_mask * xp.sin(2 * np.pi * (fx*xpp + fy*ypp)/Nact) )
    
    if return_fs:
        return xp.array(fourier_modes), sampled_fs
    else:
        return xp.array(fourier_modes)

def create_fourier_probes(
        dm_mask, 
        npsf, 
        psf_pixelscale_lamD, 
        iwa, 
        owa, 
        rotation=0, 
        fourier_sampling=0.75, 
        shifts=None, nprobes=2,
        use_weighting=False, 
    ): 
    Nact = dm_mask.shape[0]

    cos_modes, fs = create_fourier_modes(
        dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, rotation,
        fourier_sampling=fourier_sampling, 
        return_fs=True,
        which='cos',
    )

    sin_modes = create_fourier_modes(
        dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, rotation,
        fourier_sampling=fourier_sampling, 
        which='sin',
    )

    nfs = fs.shape[0]

    probes = xp.zeros((nprobes, Nact, Nact))
    if use_weighting:
        fmax = xp.max(np.sqrt(fs[:,0]**2 + fs[:,1]**2))
        sum_cos = 0
        sum_sin = 0
        for i in range(nfs):
            f = np.sqrt(fs[i][0]**2 + fs[i][1]**2)
            weight = f/fmax
            sum_cos += weight*cos_modes[i]
            sum_sin += weight*sin_modes[i]
        sum_cos = sum_cos
        sum_sin = sum_sin
    else:
        sum_cos = cos_modes.sum(axis=0)
        sum_sin = sin_modes.sum(axis=0)
    
    # nprobes=2 will give one probe that is purely the sum of cos and another that is the sum of sin
    cos_weights = np.linspace(1,0,nprobes)
    sin_weights = np.linspace(0,1,nprobes)
    
    shifts = [(0,0)]*nprobes if shifts is None else shifts

    for i in range(nprobes):
        probe = cos_weights[i]*sum_cos + sin_weights[i]*sum_sin
        probe = xcipy.ndimage.shift(probe, (shifts[i][1], shifts[i][0]))
        probes[i] = probe/xp.max(probe)

    return probes

def make_f(h=10, w=6, shift=(-1,0), Nact=34):
    f_command = xp.zeros((Nact, Nact))

    top_row = Nact//2 + h//2 + shift[1]
    mid_row = Nact//2 + shift[1]
    row0 = Nact//2 - h//2 + shift[1]

    col0 = Nact//2 - w//2 + shift[0] + 1
    right_col = Nact//2 + w//2 + shift[0] + 1

    rows = xp.arange(row0, top_row)
    cols = xp.arange(col0, right_col)

    f_command[rows, col0] = 1
    f_command[top_row,cols] = 1
    f_command[mid_row,cols] = 1
    return f_command

def make_ring(rad=15, Nact=34, thresh=1/2):
    y,x = (xp.indices((Nact, Nact)) - Nact//2 + 1/2)
    r = xp.sqrt(x**2 + y**2)
    ring = (rad-thresh<r) * (r < rad+thresh)
    ring = ring.astype(float)
    return ring

def make_fourier_command(x_cpa=10, y_cpa=10, Nact=34, phase=0):
    # cpa = cycles per aperture
    # max cpa must be Nact/2
    if x_cpa>Nact/2 or y_cpa>Nact/2:
        raise ValueError('The cycles per aperture is too high for the specified number of actuators.')
    y,x = xp.indices((Nact, Nact)) - Nact//2
    fourier_command = xp.cos(2*np.pi*(x_cpa*x + y_cpa*y)/Nact + phase)
    return fourier_command

def make_cross_command(xc=[0], yc=[0], Nact=34):
    y,x = (xp.indices((Nact, Nact)) - Nact//2 + 1/2)
    cross = xp.zeros((Nact,Nact))
    for i in range(len(xc)):
        cross[(xc[i]-0.5<=x) & (x<xc[i]+0.5)] = 1
        cross[(yc[i]-0.5<=y) & (y<yc[i]+0.5)] = 1
    # cross
    return cross



