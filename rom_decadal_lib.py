import matplotlib.pyplot as plt

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
    
    A = np.linalg.lstsq(y.data,
                        dy.data, 
                        rcond=None
                       )[0]
    
    eigs = np.linalg.eigvals(A)[...,0]
    #print(eigs)
    
    out_dims = None
    out_coords = None
    if len(y.dims[:-2])>0:
        out_dims = y.dims[:-2]
        out_coords = y.coords.copy
        out_coords=out_coords.pop('time').pop('v')
    
    return xr.DataArray(eigs,
                        name='BWJ',
                        dims=out_dims, 
                        coords=out_coords)