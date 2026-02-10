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
    
rom_dict = {}
rom_dict['linear_2D'] = basic_ROM

def pseudo_run( data_origin,
                T_variable,
                h_variable,
                rom_type,
                shape_name = '', # If you change anything below here, change shape_name
                inits = None,
                data_isel = {'time':slice(120,None)},
                run_len = 3600,
                members = (40,),
                ):

    # figure out file naming
    out_filename = 'pseudo_runs/'+data_origin+'_'+T_variable+'_'+h_variable+'_'+rom_type+'_'+shape_name+'.nc'
    if os.path.isfile(out_filename):
        warnings.warn('overwriting '+out_filename)

    # load indices
    indices = xr.open_dataset('indices/'+data_origin+'_indices.nc').isel(data_isel)
    T = declim(indices[T_variable])
    h = declim(indices[h_variable])

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
        dims=tuple('M'+str(i) for i in range(len(members)))+('time','v')
                         )

    # for loop through _time_ only, 
    if not (inits is None):
        out[...,0,:] = inits
    for i in range(1,run_len):
        out[...,i,:] = out[...,i-1,:]+rom.ddt_xr(out[...,i-1,:])

    # save pseudo data
    out.to_netcdf(out_filename)
    
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