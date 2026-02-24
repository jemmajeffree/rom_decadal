# Calculate T and h from large ensemble data
import numpy as np
import xarray as xr
import glob
import re
import os

from dask.distributed import Client

def average_equatorial_band_regridded(x, lat_var = 'lat',lat_bound = 5):
    # Throw out time_bounds or whatever
    if 'time_bnds' in x.data_vars:
        x = x.drop_vars('time_bnds')
        
    # Weight by latitude. Assume even spacing in latitude over this boundary
    area_weight = np.cos(np.deg2rad(x[lat_var]))*(~np.isnan(x))
    
    return ((x*area_weight).sel({lat_var:slice(-lat_bound,lat_bound)}).sum(lat_var)/
            area_weight.sel({lat_var:slice(-lat_bound,lat_bound)}).sum(lat_var))

def average_index_regridded(x,weight,lon_bound,lon_var='lon'):
    return ((x*weight).sel({lon_var:lon_bound}).sum(lon_var)/
            weight.sel({lon_var:lon_bound}).sum(lon_var))


if __name__ == '__main__':
    client = Client(threads_per_worker=1)

    # Where I'm putting these things once calculated
    out_dir = '/g/data/x77/jj8842/ENSO ROM/rom_decadal/indices/'

    # Where the data currently sits
    MMLEA_dir = '/g/data/su28/MMLEAv2/ocean/monthly/{var}/'
    filepaths = {'ACCESS-ESM1-5':'ACCESS-ESM1-5/{var}_Omon_ACCESS-ESM1-5_historical_r{M}i1p1f1_g025_185001-201412.nc',
                'CESM1-CAM5':'CESM1-CAM5/{var}_Omon_CESM1-CAM5_historical_rcp85_r{M}i1p1_g025_192001-210012.nc',
                'CESM2':'CESM2/{var}_Omon_CESM2_cmip6_historical_ssp370_{M}i1p1f1_g025_185001-210012.nc', # Some filename stuff needs sorting out
                'CanESM5':'CanESM5/{var}_Omon_CanESM5_historical_r{M}i1p2f1_g025_185001-201412.nc',
                'EC-Earth3':'EC-Earth3/{var}_Omon_EC-Earth3_historical_r{M}i1p1f1_g025_185001-201412.nc',
                'IPSL-CM6A-LR':'IPSL-CM6A-LR/{var}_Omon_IPSL-CM6A-LR_historical_r{M}i1p1f1_g025_185001-201412.nc',
                'MIROC6':'MIROC6/{var}_Omon_MIROC6_historical_r{M}i1p1f1_g025_185001-201412.nc',
                'MPI-ESM1-2-LR':'MPI-ESM1-2-LR/{var}_Omon_MPI-ESM1-2-LR_historical_r{M}i1p1f1_g025_185001-201412.nc',
                }
    
    variables = ('tos','z20') # This is only ever allowed to be two variables, I just don't
                              # want to risk accidentally typing zos out of habit
                              # If for some reason you want to use this code for more variables, the 
                              # only issue will be shared_ensemble_members checking

    
    for model in filepaths:
        if os.path.isfile(out_dir+model+'_SMILE_indices.nc'):
            print('Skipping '+out_dir+model+'_SMILE_indices.nc')
            continue
        
        # Do a huge amount of messing around just to find what ensemble members we have for both z20 and tos
        
        ensemble_members = {}
        
        # Find ensemble members for tos and z20
        for var in variables:
            matching_paths = glob.glob((MMLEA_dir+filepaths[model]).format(var=var,M='*'))
            regex = re.compile(filepaths[model].format(var=var,M='(.*)'))
            ensemble_members[var] = regex.findall('\n'.join(matching_paths))
    
        # Throw out any where we don't have both tos and z20
        shared_ensemble_members = [m for m in ensemble_members[variables[0]] 
                                   if m in ensemble_members[variables[1]]
                                  ] # List comprehension is eldrich

        if model == 'CESM2':
            # Minor quirk of the way files seem to be named, which I'm unfortunately not in a position to fix
            shared_ensemble_members = ensemble_members['tos'] 
    
        argsort = np.argsort(np.array(shared_ensemble_members,float))
        shared_ensemble_members = np.array(shared_ensemble_members)[argsort]
    
        shared_filepaths = [
            [(MMLEA_dir+filepaths[model]).format(var=var,M=m) 
             for m in shared_ensemble_members]
            for var in variables]

        if model == 'CESM2':
            add_r = {'z20':'r','tos':''}
            shared_filepaths = [
                                [(MMLEA_dir+filepaths[model]).format(var=var,M=add_r[var]+m) 
                                 for m in shared_ensemble_members]
                                for var in variables]

        # Okay, now we can actually open the data
        
        ds = xr.open_mfdataset(shared_filepaths,
                      combine = 'nested',
                      concat_dim = [None,'M'],
                      parallel = True,
                      join = 'outer',
                      coords='minimal',
    
                     chunks = {'lat':-1,'lon':-1,'time':120},
                     preprocess = average_equatorial_band_regridded, # First pass trim to equator to save time
                     ).assign_coords({'M':shared_ensemble_members}).load().squeeze()
    
        ds.to_netcdf(out_dir+model+'_SMILE_eq-ave.nc') # May as well save that

        # Then, average again to get the specific indices I actually want to use
        if model == 'CESM2':
            is_ocean = xr.Dataset({'tos':~np.isnan(xr.open_mfdataset(shared_filepaths[0][0]).tos.isel(time=0).load())}) # yeah, I'm not happy about the shifting of types either but I think it's the least worst choice in this situation
        else:
            is_ocean = ~np.isnan(xr.open_mfdataset(shared_filepaths[0][0]).isel(time=0).load())
        weight = average_equatorial_band_regridded(is_ocean).tos
        
        indices = {}
        
        indices['nino3'] = average_index_regridded(ds.tos,weight,slice(210,270))
        indices['nino34'] = average_index_regridded(ds.tos,weight,slice(190,240))
        indices['nino4'] = average_index_regridded(ds.tos,weight,slice(160,210))
        
        indices['h_eq'] = average_index_regridded(ds.z20,weight,slice(120,280)) #Izumo2019
        indices['h_w'] = average_index_regridded(ds.z20,weight,slice(120,205)) # Izumo2019

        # Save that
        out = xr.Dataset(indices)
        if model == 'CESM2': # again. C'mon
            out = out.squeeze().drop_vars('z_t')
        out.to_netcdf(out_dir+model+'_SMILE_indices.nc')                         
    
