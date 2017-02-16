import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from geoplots import mapplot
from sklearn.decomposition import PCA
from xlearn.decomposition import PCA as PCAx

from eofs.xarray import Eof
from eofs.examples import example_data_path


# Read SST anomalies using the xarray module. The file contains November-March
# averages of SST anomaly in the central and northern Pacific.
filename = example_data_path('sst_ndjfm_anom.nc')
sst = xr.open_dataset(filename)['sst']
sst -= sst.mean('time')

# Create an EOF solver to do the EOF analysis. Square-root of cosine of
# latitude weights are applied before the computation of EOFs.
coslat = np.cos(np.deg2rad(sst.coords['latitude'].values))
wgts = np.sqrt(coslat)[..., np.newaxis]
solver = Eof(sst, weights=wgts)

# Retrieve the leading EOF, expressed as the correlation between the leading
# PC time series and the input SST anomalies at each grid point, and the
# leading PC time series itself.
eof1 = solver.eofs(neofs=1)
pc1 = solver.pcs(npcs=1, pcscaling=1)

fig, axes = plt.subplots(1,2)
plt.sca(axes[0])
eof1.sel(mode=0).plot.contourf(levels=np.arange(-0.15,0.16,0.03))
mapplot()
plt.sca(axes[1])
pc1.sel(mode=0).plot()
plt.suptitle('Results from eofs package')
plt.tight_layout()

# use pca analysis from sklearn
n_samples = sst.shape[0]
grid_shape = sst.shape[1:]
n_grids = np.prod(grid_shape)

sst_ = sst.data.reshape((n_samples, n_grids))
valid_grids = ~np.isnan(sst_[0,:])
sst_ = sst_[:, valid_grids]
model = PCA(n_components=1).fit(sst_)
eof1_ = np.empty((model.n_components, n_grids)) * np.nan
eof1_[:, valid_grids] = model.components_
eof1_ = eof1_.reshape((model.n_components,)+grid_shape)
eof1_da = xr.DataArray(eof1_, dims=('mode',)+sst.dims[1:], coords=[np.array([0]),
    sst['latitude'], sst['longitude']])
fig, axes = plt.subplots(1, 2)
plt.sca(axes[0])
eof1_da.sel(mode=0).plot.contourf(levels=np.arange(-0.15,0.16,0.03))
mapplot()
plt.sca(axes[1])
plot(model.transform(sst_))
plt.suptitle('Results from sklearn PCA: no weight')
plt.tight_layout()

# xlearn PCA
result = PCAx(n_components=1).fit(sst)
fig, axes = plt.subplots(1, 2)
plt.sca(axes[0])
result.components_da.sel(mode=0).plot.contourf(levels=np.arange(-0.15,0.16,0.03))
mapplot()
plt.sca(axes[1])
result.transform(sst).sel(mode=0).plot()
plt.suptitle('Results from xlearn PCA: no weight')
plt.tight_layout()

# xlearn PCA with weight
weight = np.sqrt( np.cos( sst['latitude']*np.pi/180 ) ) + sst['longitude']*0
result = PCAx(n_components=1, weight=weight).fit(sst)
fig, axes = plt.subplots(1, 2)
plt.sca(axes[0])
result.components_da.sel(mode=0).plot.contourf(levels=np.arange(-0.15,0.16,0.03))
mapplot()
plt.sca(axes[1])
result.transform(sst).sel(mode=0).plot()
plt.suptitle('Results from xlearn PCA: with weight')
plt.tight_layout()