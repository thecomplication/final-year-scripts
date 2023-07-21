# %%
import xarray as xr

# %%
import seaborn as sns

# %%
import matplotlib.pyplot as plt
import pandas as pd

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# Open the netCDF file and convert the precipitation data into daily values
ds_data = xr.open_dataset('drive/MyDrive/afonly.nc')
da_RR = ds_data['pr'].values * 86400
da_RR = xr.DataArray(da_RR, dims=('time', 'lat', 'lon'), coords={'time': ds_data.time, 'lat': ds_data.lat, 'lon': ds_data.lon})

# %%
# Define the dry_spell() function and call it
def dry_spell(data, window=5, Rday_thresh=1, dim='time'):
    rolling_sums = data.rolling(time=window).sum()
    dry_day_count = data.where(data < Rday_thresh).rolling(time=window).count()
    spells = data.where((rolling_sums < window*Rday_thresh) & (dry_day_count == window))
    return spells

# %%
# Calculate the dry spells for different window sizes
spells_5 = dry_spell(da_RR, window=5, Rday_thresh=1, dim='time')
spells_10 = dry_spell(da_RR, window=10, Rday_thresh=1, dim='time')
spells_15 = dry_spell(da_RR, window=15, Rday_thresh=1, dim='time')
spells_20 = dry_spell(da_RR, window=20, Rday_thresh=1, dim='time')

# %%
%%time
# Count the number of dry spells per year using groupby
spells_5_count = spells_5.groupby('time.year').count(dim='time')
spells_10_count = spells_10.groupby('time.year').count(dim='time')
spells_15_count = spells_15.groupby('time.year').count(dim='time')
spells_20_count = spells_20.groupby('time.year').count(dim='time')

# %%
def spell_5_count_M(data):
    spell_5_count_M = data.resample(time='1M').count(dim='time').mean(['lon', 'lat'])
    spell_5_count_M = spell_5_count_M.to_dataframe(name='data')
    spell_5_count_M['Date'] = pd.to_datetime(spell_5_count_M.index.get_level_values('time'))
    spell_5_count_M['Year'] = spell_5_count_M['Date'].dt.year
    spell_5_count_M['Month'] = spell_5_count_M['Date'].dt.month
    spell_5_count_M = spell_5_count_M.pivot_table(values='data', index='Year', columns='Month')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    spell_5_count_M.columns = month_names

    return spell_5_count_M

def spell_10_count_M(data):
    spell_10_count_M = data.resample(time='1M').count(dim='time').mean(['lon', 'lat'])
    spell_10_count_M = spell_10_count_M.to_dataframe(name='data')
    spell_10_count_M['Date'] = pd.to_datetime(spell_10_count_M.index.get_level_values('time'))
    spell_10_count_M['Year'] = spell_10_count_M['Date'].dt.year
    spell_10_count_M['Month'] = spell_10_count_M['Date'].dt.month
    spell_10_count_M = spell_10_count_M.pivot_table(values='data', index='Year', columns='Month')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    spell_10_count_M.columns = month_names

    return spell_10_count_M

def spell_15_count_M(data):
    spell_15_count_M = data.resample(time='1M').count(dim='time').mean(['lon', 'lat'])
    spell_15_count_M = spell_15_count_M.to_dataframe(name='data')
    spell_15_count_M['Date'] = pd.to_datetime(spell_15_count_M.index.get_level_values('time'))
    spell_15_count_M['Year'] = spell_15_count_M['Date'].dt.year
    spell_15_count_M['Month'] = spell_15_count_M['Date'].dt.month
    spell_15_count_M = spell_15_count_M.pivot_table(values='data', index='Year', columns='Month')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    spell_15_count_M.columns = month_names

    return spell_15_count_M

def spell_20_count_M(data):
    spell_20_count_M = data.resample(time='1M').count(dim='time').mean(['lon', 'lat'])
    spell_20_count_M = spell_20_count_M.to_dataframe(name='data')
    spell_20_count_M['Date'] = pd.to_datetime(spell_20_count_M.index.get_level_values('time'))
    spell_20_count_M['Year'] = spell_20_count_M['Date'].dt.year
    spell_20_count_M['Month'] = spell_20_count_M['Date'].dt.month
    spell_20_count_M = spell_20_count_M.pivot_table(values='data', index='Year', columns='Month')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    spell_20_count_M.columns = month_names

    return spell_20_count_M

# Call the functions with your data
result_5 = spell_5_count_M(spells_5)
result_10 = spell_10_count_M(spells_10)
result_15 = spell_15_count_M(spells_15)
result_20 = spell_20_count_M(spells_20)




# %%
result_5

# %%
# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# Plot result_5
axes[0,0].boxplot(result_5, showfliers=False, patch_artist=True, labels=month_names)
axes[0,0].set_title('Five-day Monthly Spell Counts')
axes[0,0].set_xlabel('Month')
axes[0,0].set_ylabel('Spell Count')

# Plot result_10
axes[0,1].boxplot(result_10.values, showfliers=False, patch_artist=True, labels=month_names)
axes[0,1].set_title('Ten-day Monthly Spell Counts')
axes[0,1].set_xlabel('Month')
axes[0,1].set_ylabel('Spell Count')

# Plot result_15
axes[1,0].boxplot(result_15.values, showfliers=False, patch_artist=True, labels=month_names)
axes[1,0].set_title('Fiftheen-day Monthly Spell Counts')
axes[1,0].set_xlabel('Month')
axes[1,0].set_ylabel('Spell Count')

# Plot result_20
axes[1,1].boxplot(result_10.values, showfliers=False, patch_artist=True, labels=month_names)
axes[1,1].set_title('Twenty-day Monthly Spell Counts')
axes[1,1].set_xlabel('Month')
axes[1,1].set_ylabel('Spell Count')

# Adjust spacing between subplots
plt.tight_layout()
plt.suptitle('Monthly Spell Counts')
#
# Show the plot
#plt.show()

# %%
def spell_5_count_region(data):
    spells_5_count_West=data.sel(lon=slice(-2.4604,2.4604),lat=slice(-13.5317,13.5317))
    vw=spells_5_count_West.mean(['lon', 'lat'])
    spells_5_count_East=data.sel(lon=slice(-1.9577,1.9577),lat=slice(-37.2972,37.2972))
    ve=spells_5_count_East.mean(['lon', 'lat'])
    spells_5_count_North=data.sel(lon=slice(-32.2778,32.2778),lat=slice(-26.0198,26.0198))
    vn=spells_5_count_North.mean(['lon', 'lat'])
    spells_5_count_South=data.sel(lon=slice(-19.5687,19.5687),lat=slice(-24.3571,24.3571))
    vs=spells_5_count_South.mean(['lon', 'lat'])
    data = [vw,ve,vs,vn]
    #month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


    return spell_5_count_region

# %%
"""

"""

# %%
ddd = spell_5_count_region(spells_5_count)
fig, ax = plt.subplots()

# Create the box plot
ax.boxplot(ddd)

# Set the x-axis tick labels
ax.set_xticklabels(['Category 1', 'Category 2', 'Category 3','sd'])

# Set the labels and title
ax.set_xlabel('Category')
ax.set_ylabel('Value')
ax.set_title('Box Plot of Categories')

# Show the plot
plt.show()

# %%
#spells_10.sel(time=slice('2024','2026')).plot()

# %%
#spells_5.groupby('time.year').sum('time').plot(col='year', col_wrap=5, cmap='RdBu', vmax=20)

# %%
#spells_10.groupby('time.year').sum('time').plot(col='year', col_wrap=5, cmap='RdBu', vmax=20)

# %%
#spells_15.groupby('time.year').sum('time').plot(col='year', col_wrap=5, cmap='RdBu', vmax=20)

# %%
#spells_20.groupby('time.year').sum('time').plot(col='year', col_wrap=5, cmap='RdBu', vmax=20)

# %%
#spells_5_count.plot(col='year', col_wrap=5, cmap='RdBu_r', vmax=300)

# %%
spells_5_count_West=spells_5_count.sel(lon=slice(-2.4604,2.4604),lat=slice(-13.5317,13.5317))
vw=spells_5_count_West.mean(['lon', 'lat'])

# %%
#sns.boxplot(vw.values)

# %%
spells_5_count_East=spells_5_count.sel(lon=slice(-1.9577,1.9577),lat=slice(-37.2972,37.2972))
ve=spells_5_count_East.mean(['lon', 'lat'])

# %%
#sns.boxplot(ve.values)

# %%
spells_5_count_North=spells_5_count.sel(lon=slice(-32.2778,32.2778),lat=slice(-26.0198,26.0198))
vn=spells_5_count_North.mean(['lon', 'lat'])

# %%
#sns.boxplot(vn.values)

# %%
spells_5_count_South=spells_5_count.sel(lon=slice(-19.5687,19.5687),lat=slice(-24.3571,24.3571))
vs=spells_5_count_South.mean(['lon', 'lat'])

# %%
#sns.boxplot(vs.values)

# %%
spells_10_count.plot(col='year', col_wrap=5, cmap='RdBu_r', vmax=300)

# %%
spells_10_count_West=spells_10_count.sel(lon=slice(-2.4604,2.4604),lat=slice(-13.5317,13.5317))
xw=spells_10_count_West.mean(['lon', 'lat'])

# %%
sns.boxplot(xw.values)

# %%
spells_10_count_East=spells_10_count.sel(lon=slice(-1.9577,1.9577),lat=slice(-37.2972,37.2972))
xe=spells_10_count_East.mean(['lon', 'lat'])

# %%
sns.boxplot(xe.values)

# %%
spells_10_count_North=spells_10_count.sel(lon=slice(-32.2778,32.2778),lat=slice(-26.0198,26.0198))
xn=spells_10_count_North.mean(['lon', 'lat'])

# %%
sns.boxplot(xn.values)

# %%
spells_10_count_South=spells_10_count.sel(lon=slice(-19.5687,19.5687),lat=slice(-24.3571,24.3571))
xs=spells_10_count_South.mean(['lon', 'lat'])

# %%
sns.boxplot(xs.values)

# %%
spells_15_count.plot(col='year', col_wrap=5, cmap='RdBu_r', vmax=300)

# %%
spells_15_count_West=spells_15_count.sel(lon=slice(-2.4604,2.4604),lat=slice(-13.5317,13.5317))
xvw=spells_15_count_West.mean(['lon', 'lat'])

# %%
sns.boxplot(xvw.values)

# %%
spells_15_count_East=spells_15_count.sel(lon=slice(-1.9577,1.9577),lat=slice(-37.2972,37.2972))
xve=spells_15_count_East.mean(['lon', 'lat'])

# %%
sns.boxplot(xve.values)

# %%
spells_15_count_North=spells_15_count.sel(lon=slice(-32.2778,32.2778),lat=slice(-26.0198,26.0198))
xvn=spells_15_count_North.mean(['lon', 'lat'])

# %%
sns.boxplot(xvn.values)

# %%
spells_15_count_South=spells_15_count.sel(lon=slice(-19.5687,19.5687),lat=slice(-24.3571,24.3571))
xvs=spells_15_count_South.mean(['lon', 'lat'])

# %%
sns.boxplot(xvs.values)

# %%
spells_20_count.plot(col='year', col_wrap=5, cmap='RdBu_r', vmax=300)

# %%
spells_20_count_West=spells_20_count.sel(lon=slice(-2.4604,2.4604),lat=slice(-13.5317,13.5317))
xxw=spells_20_count_West.mean(['lon', 'lat'])

# %%
sns.boxplot(xxw.values)

# %%
spells_20_count_East=spells_20_count.sel(lon=slice(-1.9577,1.9577),lat=slice(-37.2972,37.2972))
xxe=spells_20_count_East.mean(['lon', 'lat'])

# %%
sns.boxplot(xxe.values)

# %%
spells_20_count_North=spells_20_count.sel(lon=slice(-32.2778,32.2778),lat=slice(-26.0198,26.0198))
xxn=spells_20_count_North.mean(['lon', 'lat'])

# %%
sns.boxplot(xxn.values)

# %%
spells_20_count_South=spells_20_count.sel(lon=slice(-19.5687,19.5687),lat=slice(-24.3571,24.3571))
xxs=spells_20_count_South.mean(['lon', 'lat'])

# %%
sns.boxplot(xxs.values)

# %%
# Select a single latitude and longitude location to plot
lat_idx = 0
lon_idx = 0

# %%
spells_20_count.values

# %%
# Plot the number of dry spells per year at the selected location
fig, ax = plt.subplots()
ax.plot(spells_5_count.year, spells_5_count.mean(['lon', 'lat']), label='Window size = 5')
ax.plot(spells_10_count.year, spells_10_count.mean(['lon', 'lat']), label='Window size = 10')
ax.plot(spells_15_count.year, spells_15_count.mean(['lon', 'lat']), label='Window size = 15')
ax.plot(spells_20_count.year, spells_20_count.mean(['lon', 'lat']), label='Window size = 20')
ax.set_xlabel('Year')
ax.set_ylabel('Number of dry spells')
ax.legend()
plt.show()
plt.savefig('Zola')

# %%
# Plot the number of dry spells per year at the selected location
fig, ax = plt.subplots()
ax.plot(vw.year, vw.values, label='Western Africa')
ax.plot(ve.year, ve.values, label='Eastern Africa')
ax.plot(vn.year, vn.values, label='Northern Africa')
ax.plot(vs.year, vs.values, label='Southern Africa')
ax.set_xlabel('Years')
ax.set_ylabel('Count of dry spells')
ax.legend()
plt.show()

# %%
# Plot the number of dry spells per year at the selected location
fig, ax = plt.subplots()
ax.plot(xw.year, xw.values, label='Western Africa')
ax.plot(xe.year, xe.values, label='Eastern Africa')
ax.plot(xn.year, xn.values, label='Northern Africa')
ax.plot(xs.year, xs.values, label='Southern Africa')
ax.set_xlabel('Years')
ax.set_ylabel('Count of dry spells')
ax.legend()
plt.show()

# %%
# Plot the number of dry spells per year at the selected location
fig, ax = plt.subplots()
ax.plot(xvw.year, xvw.values, label='Western Africa')
ax.plot(xve.year, xve.values, label='Eastern Africa')
ax.plot(xvn.year, xvn.values, label='Northern Africa')
ax.plot(xvs.year, xvs.values, label='Southern Africa')
ax.set_xlabel('Years')
ax.set_ylabel('Count of dry spells')
ax.legend()
plt.show()

# %%
# Plot the number of dry spells per year at the selected location
fig, ax = plt.subplots()
ax.plot(xxw.year, xxw.values, label='Western Africa')
ax.plot(xxe.year, xxe.values, label='Eastern Africa')
ax.plot(xxn.year, xxn.values, label='Northern Africa')
ax.plot(xxs.year, xxs.values, label='Southern Africa')
ax.set_xlabel('Years')
ax.set_ylabel('Count of dry spells')
ax.legend()
plt.show()

# %%
