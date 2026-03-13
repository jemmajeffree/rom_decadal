import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
import warnings
import copy


plt.figure()
plt.rcParams.update({'font.size':16}) # Set up some defaults for all my figures

# A bunch of random plotting colours
ENC = '#FF888F' 
LNC = '#9FCCFF'
BJC = '#2e6a57'
WKC = '#f6a895'

def declim(x):
    return x.groupby('time.month')-x.groupby('time.month').mean()


def calc_BWJ(y,dy):
    ''' Calculate the Bjerknes-Wyrtki-Jin index
    (complex number Bj + i*Wk )
    '''
    y = y.transpose(..., 'time', 'v')
    dy = dy.transpose(..., 'time', 'v')
    
    A = xr.apply_ufunc(lambda y, dy: np.linalg.lstsq(y,dy,rcond=None)[0],
                       y,
                       dy,
                       input_core_dims=[('time','v')]*2,
                       output_core_dims=[('v1','v2')],
                       vectorize=True,
                      )
    
    eigs = np.linalg.eigvals(A)[...,0]
    
    out_dims = None
    out_coords = None
    if len(A.dims[:-2])>0:
        out_dims = A.dims[:-2]
        out_coords = A.coords
    
    return xr.DataArray(eigs,
                        name='BWJ',
                        dims=out_dims, 
                        coords=out_coords)

class basic_ROM:
    
    def __init__(self, A=None, sigma=None):
        ''' Just in case I ever feel like overwriting these instead of fitting. 
        Can't see it happening, tbh, but it's good to have versatile code'''
        
        self.A = A
        self.sigma = sigma
    
    def fit(self, y0, dy):
        
        y0 = y0.transpose(..., 'time', 'v')
        dy = dy.transpose(..., 'time', 'v')

        x, res2, *_ = np.linalg.lstsq(y0.data, dy.data, rcond=None)
        
        self.A = x.T # Unclear to me why the transpose is needed, but it reliably is to get the same answer back and it's not cancelling any others
        self.sigma = (res2/y0.time.shape[0])**0.5
            
    def ddt_xr(self, y0):
        ''' slower, but more robust'''
        
        # Do a bunch of dimension management
        orig_dims = copy.copy(y0.dims)
        y0 = y0.transpose(..., 'v')
        out_xr = xr.zeros_like(y0)
        y0 = y0.expand_dims('spare_dim',-1) # So that v is _always_ n-1
        
        out_xr[:] = np.squeeze(self.A@y0.data + 
                              np.random.normal(0,
                                               np.expand_dims(self.sigma,-1), # to match the other expanded dim
                                                y0.shape),
                            -1,
                            )
        
        return out_xr.transpose(*orig_dims)
    
class quadT_ROM:
    
    def __init__(self, A=None, sigma=None):
        ''' Just in case I ever feel like overwriting these instead of fitting. 
        Can't see it happening, tbh, but it's good to have versatile code'''
        
        self.A = A # A is a numpy array of shape (2,3), BUT A[1,2] = 0 because h is not dependent on T^2
        self.sigma = sigma
    
    def fit(self, y0, dy):
        
        y0 = y0.transpose(..., 'time', 'v')
        dy = dy.transpose(..., 'time', 'v')
        
        # nonlinear bit for T
        xT, res2T, *_ = np.linalg.lstsq(xr.concat((y0,y0.sel(v=np.array('T'))**2),dim='v'),
                                      dy.sel(v='T').data, 
                                      rcond=None,
                                      )

        # linear bit for h
        xh, res2h, *_ = np.linalg.lstsq(y0.data, 
                                      dy.sel(v='h').data, 
                                      rcond=None)
        
        self.A = np.zeros((2,3))
        self.A[0] = xT
        self.A[1,:2] = xh
        
        self.sigma = np.concatenate(((res2T/y0.time.shape[0])**0.5,(res2h/y0.time.shape[0])**0.5))
            
    def ddt_xr(self, y0):
        
        # Do a bunch of dimension management
        orig_dims = copy.copy(y0.dims)
        y0 = y0.transpose(..., 'v')
        out_xr = xr.zeros_like(y0)
        
        # Prep for linalg
        y0 = xr.concat((y0,y0.sel(v='T')**2),'v')
        y0 = y0.expand_dims('spare_dim',-1) # So that v is _always_ n-1
        
        out_xr[:] = np.squeeze(self.A@y0.data + # predictable bit
                              np.random.normal(0, # noise
                                               np.expand_dims(self.sigma,-1), # to match the other expanded dim
                                                y0.isel(v=slice(None,2)).shape),
                            -1,
                            )
        
        return out_xr.transpose(*orig_dims)
    
class lookup_table_ROM:
    def __init__(self, A=None, gridded_tendency = None, sigma=None):
        ''' Just in case I ever feel like overwriting these instead of fitting. 
        Can't see it happening, tbh, but it's good to have versatile code'''
        
        self.A = A # backup linear matrix
        self.gridded_tendency = gridded_tendency # lookup table
        self.sigma = sigma # error (T, h)
    
    def fit(self, y0, dy, count_threshold=5):
        
        y0 = y0.transpose(..., 'time', 'v')
        dy = dy.transpose(..., 'time', 'v')
        assert len(y0.shape)==2, "Look, sorry, but I'm not coding this to handle more dimensions"
        
        # First, get a linear fit to use as a backup when the ROM leaves the area we can get an empirical fit
        x, res2, *_ = np.linalg.lstsq(y0.data, dy.data, rcond=None)
        self.A = x.T
        
        # Now we build the lookup-table and fill it in. Going to 8 sigma with spacing of 0.5 std
        # define edges of bins
        h_bins = np.linspace(-8,8,8*4+1)
        T_bins = np.linspace(-8,8,8*4+1) # We have problems if we ever get an ENSO event > 8σ
        std = y0.std('time') # scaling factor to fit stuff in the box
        assert np.all(np.abs(y0)<8*std) # everything had better be less than 8 std
        
        h_bins = h_bins*float(std.sel(v='h'))
        T_bins = T_bins*float(std.sel(v='T'))
        
        # define empty lookup table
        gridded_tendency = np.zeros((2,T_bins.shape[0]-1,h_bins.shape[0]-1))
        variance = np.zeros((2,T_bins.shape[0]-1,h_bins.shape[0]-1))
        bin_count = np.zeros((T_bins.shape[0]-1,h_bins.shape[0]-1))
        
        for hi,h in enumerate(h_bins[:-1]):
            for Ti,T in enumerate(T_bins[:-1]):
                relevant_data = dy.where((y0.sel(v='h')>=h_bins[hi]) &
                                               (y0.sel(v='h')<h_bins[hi+1]) &
                                               (y0.sel(v='T')>=T_bins[Ti]) &
                                               (y0.sel(v='T')<T_bins[Ti+1])
                                              )
                bin_count[Ti,hi] = np.sum(~np.isnan(relevant_data))
                
                if bin_count[Ti,hi]>=count_threshold:
                    gridded_tendency[:,Ti,hi] = relevant_data.mean('time')
                    variance[:,Ti,hi] = relevant_data.var('time',ddof=0) # Technically this should be ddof=1, but I really can't be bothered explaining statistics to people who should know better and it doesn't meaningfully affect the results
                else:
                    mean_T = (T_bins[Ti]+T_bins[Ti+1])/2
                    mean_h = (h_bins[hi]+h_bins[hi+1])/2
                    gridded_tendency[:,Ti,hi] = self.A@np.array((mean_T,mean_h))
                
        self.sigma = (np.sum(variance*bin_count,axis=(1,2)) # Again, I think if we were doing this really properly we'd weight by bin_count minus 1
                      /np.sum(bin_count,axis=(0,1)))**0.5
        
        self.gridded_tendency = xr.DataArray(gridded_tendency,dims=('v','T','h'),
                                           coords={'T':(T_bins[:-1]+T_bins[1:])/2,'h':(h_bins[:-1]+h_bins[1:])/2})

            
    def ddt_xr(self, y0):
        
        # Do a bunch of dimension management
        orig_dims = copy.copy(y0.dims)
        y0 = y0.transpose(..., 'v')
        
        return (self.gridded_tendency.sel(T=y0.sel(v='T'),h=y0.sel(v='h'),method='nearest').transpose(..., 'v') # predictable bit
                +np.random.normal(0,self.sigma,y0.isel(v=slice(None,2)).shape) # noise
                ).transpose(*orig_dims)
    
rom_dict = {}
rom_dict['linear_2D'] = basic_ROM
rom_dict['quadT'] = quadT_ROM
rom_dict['lookup_table'] = lookup_table_ROM


def decimal_year(t):
    ''' Turns date into a number that can be used for detrending. May be off by 1/366 in Gregorian calendars, but 
    I think that's well within the margin of error'''
    
    return t['time.year']+t['time.dayofyear']/365

def detrend_ensemble_mean(y):
    ''' Simple "remove forced signal" via ensemble mean. Seems to produce
    a couple spurious non-oscillatory decades, and can't work on obs, so I'm
    using the quadratic instead'''
    
    assert 'M' in y.dims, 'Not gonna work without multiple ensemble members'
    return y-y.mean('M')

def declim_detrend_quadratic(y):
    ''' Remove the climate change signal in obs or a SMILE
    by removing the seasonal cycle and then fitting and subtracting
    a quadratic.
    Currently assumes that you have a semi-reasonable time unit, so 
    works best with decode_times=False'''
    # strip seasonal cycle
    y = declim(y) 
    
    # grab old time coordinates to put back later
    old_time_coords = y.time.copy() 
    
    # then put nice easy coordinates on
    # new_time_coords = (y.time-np.datetime64('1900-01-01'))/np.timedelta64(1,'D')
    new_time_coords = decimal_year(y.time)
    y = y.assign_coords({'time':new_time_coords}) 
    
    # Get polynomial coefficients
    if 'M' in y.dims:
        fit_curve = y.mean('M')
    else:
        fit_curve = y
    fit = fit_curve.polyfit(dim='time',deg=2).polyfit_coefficients
    
    return (y-                 # original minus
            (y.time**2*fit[0]+ # quadratic
              y.time*fit[1]+   # linear
              fit[2])          # constant
           ).assign_coords({'time':old_time_coords})

def pseudo_run( data_origin,
                T_variable,
                h_variable,
                rom_type,
                shape_name = '', # If you change anything below here, change shape_name
                inits = None,
                data_isel = {'time':slice(120,None)},
                data_labelsel = {},
                run_len = 3600,
                members = (40,),
                detrend = declim,
                ):

    # figure out file naming
    out_filename = 'pseudo_runs/'+data_origin+'_'+T_variable+'_'+h_variable+'_'+rom_type+'_'+shape_name+'.nc'
    if os.path.isfile(out_filename):
        warnings.warn('overwriting '+out_filename)

    # load indices
    indices = xr.open_dataset('indices/'+data_origin+'_indices.nc').isel(data_isel).sel(data_labelsel)
    T = detrend(indices[T_variable])
    h = detrend(indices[h_variable])

    full_y = xr.concat((T,h),dim='v').assign_coords({'v':np.array(('T','h'))})
    full_dy = full_y.diff('time',label='lower') # this is the dy at the time it starts from
    full_y = full_y.isel(time=slice(None,-1)) # We've got no dy for the end :)

    # Vectorisation assumes anything other than time is parallel, rather than more of the same data
    if 'M' in full_y.dims:
        full_y = full_y.rename({'time':'t'}).stack({'time':('t','M')})
        full_dy = full_dy.rename({'time':'t'}).stack({'time':('t','M')})

    # fit model
    rom = rom_dict[rom_type]()
    rom.fit(full_y,full_dy)

    # array to put pseudo data into
    out = xr.DataArray(
        np.ones(members+(run_len,)+(2,)),
        dims=tuple('M'+str(i) for i in range(len(members)))+('time','v'),
        name = 'pseudo_run',
        coords = {'v':np.array(('T','h'))},
                         )

    # for loop through _time_ only, 
    if not (inits is None):
        out[...,0,:] = inits
    for i in range(1,run_len):
        out[...,i,:] = out[...,i-1,:]+rom.ddt_xr(out[...,i-1,:])

    # save pseudo data
    out.to_netcdf(out_filename)
    indices.close()
    
def plot_edge_only_histogram(data, bins=10, edge_color='black',weights=None,density=False,linestyle='solid',linewidth=1):
    """
    Courtesy of ChatGPT (and some people on stack-overflow, I kinda had to walk chatgpt through it)
    Plots a histogram showing only the outer edges of the bars without any fills.
    
    Parameters:
        data (array-like): The input data for the histogram.
        bins (int): The number of bins for the histogram.
        edge_color (str): The color of the edges.
    """
    # Create the histogram
    counts, bin_edges = np.histogram(data, bins=bins,weights=weights,density=density)
    
    x_edges = np.concatenate(([bin_edges[0]], bin_edges[:-1], [bin_edges[-1]]))
    y_counts = np.concatenate(([0], counts, [0]))

    # Use step to create the outer edges
    plt.step(x_edges, y_counts, where='post', color=edge_color,linestyle=linestyle,linewidth=linewidth)