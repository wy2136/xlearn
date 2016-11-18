from xlearn.cluster import KMeans
import xarray as xr
import numpy as np

# parameters
ncfile = '/home/wenchay/mydata/erai/daily/phi/500mb/ydrunanom.detrend/*.nc'
lonslice = (-90, 60)
latslice = (80, 20)
level = 500
n_clusters = 4

# load and prepare data
ds = xr.open_mfdataset(ncfile)
da = ds.z.geo.roll_lon().sel(level=level,
    lon=slice(*lonslice), lat=slice(*latslice))
# assign np.NaN values to some grids
zz = da.values
zz[:, 12:17, 35:40] = np.NaN
da.values = zz

# k-means
# m = KMeans(n_clusters, random_state=0, n_jobs=-1).fit(da)
m = da.learn.kmeans(n_clusters, random_state=0, n_jobs=-1)

# plot
m.plot_cluster_centers(proj='ortho')
