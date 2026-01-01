# Calculate T and h from ORAS5 reanalysis data
import numpy as np
import xarray as xr
import glob
import re
import os

from dask.distributed import Client

from shared_functions import average_pacific_region
import shared_functions as sf

def trim_sst_loosely(x):
    return x.sosstsst.sel(y=slice(400,600))
def trim_z20_loosely(x):
    return x.so20chgt.sel(y=slice(400,600))

if __name__ == '__main__':
    client = Client(threads_per_worker=1)

    # Where I'm putting these things once calculated
    out_dir = '/g/data/x77/jj8842/ENSO ROM/rom_decadal/indices/'
    model = 'ORAS5' # for filenaming purposes

    sst = xr.open_mfdataset('/g/data/x77/jj8842/ORAS5/sosstsst*.nc',
                       chunks={'time_counter':-1,'x':-1,'y':-1},
                        preprocess=trim_sst_loosely,
                        compat='override',
                        coords='minimal',
                        join='outer',
                       parallel=True).sosstsst

    z20 = xr.open_mfdataset('/g/data/x77/jj8842/ORAS5/so20chgt*.nc',
                   chunks={'time_counter':-1,'x':-1,'y':-1},
                    preprocess=trim_z20_loosely,
                    compat='override',
                    coords='minimal',
                    join='outer',
                   parallel=True).so20chgt

    equator = sst.isel(time_counter=0).where(np.abs(sst.nav_lat.load())<5,drop=True)
    # Just make certain that this stuff is on a regular grid here
    assert np.all((equator.nav_lat.sel(x=0)-equator.nav_lat)==0)
    assert np.all((equator.nav_lon.sel(y=0)-equator.nav_lon)==0)
    # Grab weights for all the difference it makes
    weights = ~np.isnan(equator)
    weights = weights*np.cos(np.deg2rad(weights.nav_lat))

    ds = xr.Dataset({'tos':sst,'z20':z20})

    eq_ave = average_pacific_region(ds.where(np.abs(sst.nav_lat.load())<5,drop=True),
                           weights,np.array((0,360,-5,5)),'nav_lon','nav_lat',('y',))
    
    eq_ave.to_netcdf(out_dir+model+'_obs_eq-ave.nc') # May as well save that

        
    indices = {}
    shared_arguments = {'ds':ds.where(np.abs(sst.nav_lat.load())<5,drop=True).tos.load(),
                        'ave_dims':('x','y'),
                        'weight':weights,
                        'lon_coord':'nav_lon',
                        'lat_coord':'nav_lat',
                       }
        
    indices['nino3'] = average_pacific_region(region=sf.nino3_region,**shared_arguments)
    indices['nino34'] = average_pacific_region(region=sf.nino34_region,**shared_arguments)
    indices['nino4'] = average_pacific_region(region=sf.nino4_region,**shared_arguments)
        
    shared_arguments['ds'] = ds.where(np.abs(sst.nav_lat.load())<5,drop=True).z20.load()
    
    indices['h_eq'] = average_pacific_region(region=sf.wholeP_region,**shared_arguments)
    indices['h_w'] = average_pacific_region(region=sf.westP_region,**shared_arguments)


    # Save that
    xr.Dataset(indices).to_netcdf(out_dir+model+'_obs_indices.nc')                         
    
