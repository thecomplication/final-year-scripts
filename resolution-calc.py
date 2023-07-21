import xarray as xr

# Open the NetCDF file
ds = xr.open_dataset('afonly.nc')

# Check the attributes of the latitude and longitude variables
print(ds['lat'])
print(ds['lon'])
