import xarray as xr

# Load the NetCDF file
data = xr.open_dataset('afonly.nc')

# Extract the Africa region using longitude and latitude coordinates
africa = data.sel(lon=slice(27.86, 48.25), lat=slice(9.9, -21.5))

# Save the extracted region to a new NetCDF file
africa.to_netcdf('east.nc')

