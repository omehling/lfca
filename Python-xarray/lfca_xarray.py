#!/usr/bin/env python3
"""
Fast Python implementation (using xarray) of low-frequency component analysis

Reference:  R. Wills et al.: Disentangling Global Warming, Multidecadal Variability,
            and El Niño in Pacific Temperatures,
            Geophysical Research Letters 45, 2487-2496 (2018)

This code:  Oliver Mehling, 2023, based on the previous Python version by Zhaoyi Shen
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from eofs.standard import Eof
from scipy.signal import convolve, butter, sosfiltfilt

### Helper functions ###
def low_pass_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.

    Args:

    window: int
        The length of the filter window.

    cutoff: float
        The cutoff frequency in inverse time steps.
        
    References
    ----------
        from https://scitools-iris.readthedocs.io/en/v3.7.0/generated/gallery/general/plot_SOI_filtering.html

    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]

def filter_padding(ts, window, ftype='mirror', detrend=True, detrend_poly=1):
    ts_pad = np.zeros(2*window+len(ts))
    if detrend:
        t = np.arange(len(ts))
        z = np.polyfit(t,ts,detrend_poly)
        p = np.poly1d(z)
        ts_in = ts-p(t)
    else:
        ts_in = ts
    ts_pad[window:-window] = ts_in[:]
    if ftype == 'mirror':
        ts_pad[:window] = ts_in[:window][::-1]
        ts_pad[-window:] = ts_in[-window:][::-1]
    elif ftype == 'periodic':
        ts_pad[:window] = ts_in[-window:]
        ts_pad[-window:] = ts_in[:window]
    else:
        raise ValueError('in filter_padding: ftype must be one of "mirror" or "periodic".')
    if detrend:
        t_pad = np.arange(-window,len(ts)+window)
        ts_pad = ts_pad+p(t_pad)
    return ts_pad

def filter_ts(ts, cutoff, filter_type='lanczos', padding_type='periodic', detrend=True, detrend_poly=1):
    lanczos_weights=low_pass_weights(cutoff*2+1,1./cutoff) # weights for Lanczos filter
    n_pad=int(np.ceil(len(lanczos_weights)/2))
    
    # Padding
    ts_mirr=filter_padding(ts,n_pad,padding_type,detrend=detrend,detrend_poly=detrend_poly)
    
    # Filtering
    if filter_type=='lanczos':
        # Lanczos filter
        return convolve(ts_mirr,lanczos_weights,'same')[n_pad:-n_pad]
    elif filter_type=='butter':
        # 4th-order Butterworth filter
        sos = butter(4, 1./cutoff, 'lowpass', fs=1, output='sos')
        return sosfiltfilt(sos, ts_mirr-np.mean(ts_mirr))[n_pad:-n_pad]+np.mean(ts_mirr)
    else:
        raise ValueError('in filter_ts: filter_type must be one of "lanczos" or "butter".')

### Main LFCA function ###
def lfca_new(x, cutoff, truncation, weights, **kwargs):
    if x.ndim!=2:
        raise ValueError('x must have dimension 2 for LFCA')
    
    # Scale vector from weights
    scale = np.sqrt(np.transpose(weights)/np.sum(weights))
    
    # center data
    x = x - np.mean(x,axis=0)
    xs = x * scale.T
    
    # Compute EOFs using eofs package
    eofs_np=Eof(xs, center=False, ddof=1)
    # Principal component time series (PC_k)
    pcs=eofs_np.pcs(npcs=truncation, pcscaling=1)
    
    # Filtering of PCs
    pcs_filt=np.zeros(pcs.shape)
    for i in range(truncation):
        pci = pcs[:,i]
        pci_filt = filter_ts(pci, cutoff, **kwargs)
        pcs_filt[:,i]=pci_filt[:]
        #print(np.std(pci),np.std(pci_filt))
    
    # Compute low-frequency components
    cov_lowfreq=np.cov(pcs_filt,rowvar=False)
    eig_lowfreq, eigv_lowfreq = np.linalg.eigh(cov_lowfreq) # Eigenvalues r_k, Eigenvectors e_k
    eig_argsort = np.argsort(eig_lowfreq)[::-1].copy() # Guarantee that eigenvalues are sorted in descending order
    eig_lowfreq = eig_lowfreq[eig_argsort].copy()
    eigv_lowfreq = eigv_lowfreq[:,eig_argsort].copy()
    uvec=eofs_np.eofs(neofs=truncation, eofscaling=1).T@eigv_lowfreq # u_k
    lfcs=xs@uvec # Low-frequency components (LFC_k)
    lfps=eofs_np.eofs(neofs=truncation, eofscaling=2).T@eigv_lowfreq # v_k = Low-frequency patterns (LFP_k)
    # NB: Alternative formula (lfps=xs.T@lfcs) gives the right pattern, but wrong scaling
    
    # Choose positive sign of LFCs/LFPs
    for j in range(lfps.shape[1]):
        if np.dot(lfps[:,j], scale.flatten())<0:
            lfps[:,j] = -lfps[:,j]
            lfcs[:,j] = -lfcs[:,j]
    lfps=lfps/scale # Re-scale LFPs (non-weighted)
    
    return lfcs, lfps

### LFCA class ###
class Lfca:
    def __init__(self, x, weights=None, domain=None):
        self._named_dims = list(set(x.dims)-set(['time']))
        self.x = x.transpose(*(self._named_dims+['time']))
        
        # Store dimensions etc
        self._dims_shape = self.x.shape[:-1]
        self._ntotal = np.prod(np.asarray(self._dims_shape)) # size of flattened arrays
        self._ntime = self.x.shape[-1] # size of time axis
        
        # Format input fields
        if weights is not None:
            self.weights = weights.transpose(*self._named_dims)
        else:
            self.weights = self.__ones()
        
        if domain is not None:
            self.domain = domain.transpose(*self._named_dims)
        else:
            self.domain = self.__ones()
        
        # Store remaining arguments
        self._order = 'C' # argument for (back-)transform
    
    # Private (helper) functions
    def __ones(self):
        coords_dict = {}
        for dim in self._named_dims:
            coords_dict[dim] = self.x[dim]
        return xr.DataArray(data=np.ones(self._dims_shape), coords=coords_dict)
    
    def prepare_lfca(self):
        # Pre-processing before LFCA: flatten and mask all arrays
        x_flat = np.reshape(self.x.values,(self._ntotal,self._ntime),order=self._order).T
        weights_flat = np.reshape(self.weights.values,(self._ntotal,1),order=self._order).T
        domain_flat = np.reshape(self.domain.values,(self._ntotal,),order=self._order).T

        # Mask out cells with (a) domain value 0, (b) weight 0, (c) normalized variance 0
        mask_inv = np.logical_or(
                        np.logical_or(np.isclose(domain_flat,0), np.isclose(weights_flat[0,:],0)),
                        np.logical_or(np.isclose(np.std(x_flat,axis=0)/np.std(x_flat),0), np.any(np.isnan(x_flat),axis=0))
                   )
        self._idx_retain = np.where(~mask_inv)[0]
        self._idx_discard = np.where(mask_inv)[0]
        x_flat_in = x_flat[:,self._idx_retain].copy()
        weights_flat_in = weights_flat[:,self._idx_retain].copy()
        
        return x_flat_in, weights_flat_in
    
    def lfca(self, cutoff, truncation, **kwargs):
        # Pre-process
        x_flat_in, weights_flat_in = self.prepare_lfca()
        # Run
        lfcs_np, lfps_np = lfca_new(x_flat_in, cutoff, truncation, weights_flat_in, **kwargs)
        # Post-process
        lfcs_xr, lfps_xr = self.process_lfca(lfcs_np, lfps_np)
        self.lfcs = lfcs_xr
        self.lfps = lfps_xr
        self.cutoff = cutoff
        self.truncation = truncation

        return lfcs_xr, lfps_xr
    
    def process_lfca(self, lfcs, lfps):
        # Post-processing after LFCA: back-transform arrays to original shape
        nrows = lfps.shape[1]
        
        # LFPs
        lfps_aug = np.zeros((nrows,self._ntotal))
        lfps_aug[:] = np.nan
        lfps_aug[:,self._idx_retain] = lfps.T
        patterns = np.zeros(tuple([nrows]+list(self._dims_shape)))
        for i in range(nrows):
            pattern = np.reshape(lfps_aug[i,...],self._dims_shape,order=self._order)
            #pattern[np.where(np.abs(pattern)>1.e5)] = np.nan
            patterns[i,...] = pattern
        coords_dict = {'mode':np.arange(nrows)}
        for dim in list(self._named_dims):
            coords_dict[dim] = self.x[dim]
        lfps_xr = xr.DataArray(data=patterns,coords=coords_dict)
        
        # LFCs
        lfcs_xr=xr.DataArray(
            data=lfcs,
            coords={'time':self.x.time,'mode':np.arange(nrows)}
        )
        return lfcs_xr, lfps_xr
    
    def plot_lfc(self, i, x_coord=None, cbar_scale=None):
        plot_pattern = self.lfps.sel(mode=i)
        plot_lfc = self.lfcs.sel(mode=i)
        fig,ax = plt.subplots(2,1,figsize=(6,6),gridspec_kw={'height_ratios':[2,1]})
        if cbar_scale is None:
            cbar_scale=max(np.abs(plot_pattern.max().item()), np.abs(plot_pattern.min().item()))
        c=plot_pattern.plot(x=x_coord,ax=ax[0],vmin=-cbar_scale,vmax=cbar_scale,cmap='RdYlBu_r')
        #c=ax[0].pcolormesh(np.squeeze(lon_axis),np.squeeze(lat_axis),np.transpose(pattern),
        #                vmin=-cbar_scale,vmax=cbar_scale,cmap='RdYlBu_r') #np.arange(-1,1.1,0.1)
        #fig.colorbar(c,ax=ax[0])
        ##ax[1].fill_between(plot_lfc.time,plot_lfc.values,color='#bbbbbb')
        plot_lfc.plot(ax=ax[1],lw=.3,c='C7')
        #ax[1].plot(time,lfcs[:,i],lw=.3,c='C7')
        plot_lfc.rolling(time=self.cutoff, center=True).mean('time').plot(c='k',lw=1)
        ax[1].axhline(0,lw=.5,c='C7')
        #ax[1].plot(time,pd.Series(lfcs[:,i]).rolling(72,center=True).mean().values,c='k',lw=1)
        ax[0].set(title='LFC {}'.format(i+1),xlabel='',ylabel='')
        ax[1].set(title='',xlabel='Year')

        return fig, ax
