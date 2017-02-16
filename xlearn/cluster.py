'''author: Wenchang Yang (yang.wenchang@uci.edu)'''

import xarray as xr
from sklearn.cluster import KMeans as KMeans_sklearn
import numpy as np
import matplotlib.pyplot as plt

class KMeans(KMeans_sklearn):
    '''Inherit from sklearn.cluster.KMeans that accept xarray.DataArray data
    to fit.'''

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
    def __str__(self):
        attrs = ['algorithm', 'copy_x', 'init', 'max_iter',
            'n_clusters', 'n_init', 'n_jobs', 'precompute_distances',
            'random_state', 'tol', 'verbose']
        # s = ', '.join(['{}={}'.format(a, getattr(self, a)) for a in attrs])
        values = []
        for attr in attrs:
            value = getattr(self, attr)
            if isinstance(value, str):
                value = '"{}"'.format(value)
            values.append(value)
        s = ', '.join(['{}={}'.format(attr, value)
            for attr,value in zip(attrs, values)])
        return '[KMeans for xarray]: {}.'.format(s)
    def __repr__(self):
        return self.__str__()

    def fit(self, da):
        '''xarray.DataArray version of sklearn.cluster.KMeans.fit.'''

        # Compatible with the slearn.cluster.KMeans fit method when the input data is not DataArray
        if not isinstance(da, xr.DataArray):
            return super().fit(da)

        # retrieve parameters
        n_samples = da.shape[0]
        samples_dim = da.dims[0]
        samples_coord = {samples_dim: da.coords[samples_dim]}
        features_shape = da.shape[1:]
        n_features = np.prod(features_shape)
        features_dims = da.dims[1:]
        cluster_centers_dims = ('cluster',) + features_dims
        cluster_centers_coords = {'cluster': np.arange(self.n_clusters)}
        for dim in features_dims:
            cluster_centers_coords[dim] = da.coords[dim]

        # transform the input data array
        X = da.data.reshape(n_samples, n_features)# 'data' might be replaced with 'values'.
        # any feature contains np.NaN
        valid_features_index = ~np.isnan(X[0,:])
        X_valid = X[:, valid_features_index]
        super().fit(X_valid)
        self.valid_features_index_ = valid_features_index

        # wrap the estimated parameters into DataArray
        cluster_centers = np.zeros((self.n_clusters, n_features)) + np.NaN
        cluster_centers[:, valid_features_index] = self.cluster_centers_
        self.cluster_centers_da = xr.DataArray(
            cluster_centers.reshape((self.n_clusters,) + features_shape),
            dims=cluster_centers_dims,
            coords=cluster_centers_coords
        )
        self.labels_da = xr.DataArray(self.labels_, dims=samples_dim,
            coords=samples_coord)

        return self

    def predict(self, da):
        '''xarray.DataArray version of sklearn.cluster.KMeans.fit.'''

        # compatible with the sklean.cluster.KMeans predict method when the input data is not DataArray
        if not isinstance(da, xr.DataArray):
            return super().predict(da)

        # retrieve parameters
        n_samples = da.shape[0]
        features_shape = da.shape[1:]
        n_features = np.prod(features_shape)

        X = da.data.reshape(n_samples, n_features)# 'data' might be replaced with 'values'.
        # remove NaN values if exists in X
        try:
            X_valid = X[:, self.valid_features_index_]
        except:
            X_valid = X
        samples_dim = da.dims[0]
        samples_coord = {samples_dim: da.coords[samples_dim]}
        labels = xr.DataArray(super().predict(X_valid),
            dims=samples_dim, coords=samples_coord)

        return labels

    def get_cluster_fraction(self, label):
        '''Get the fraction of a given cluster denoted by label.'''
        return (self.labels_==label).sum()/self.labels_.size

    def plot_cluster_centers(self, label=None, **kw):
        '''Plot maps of cluster centers.'''
        if label is None:
            label = range(self.n_clusters)
        elif isinstance(label, int):
            label = [label,]
        for label_ in label:
            plt.figure()
            try:
                # if the geoplots package is installed
                self.cluster_centers_da.sel(cluster=label_).geo.plot(**kw)
            except:
                self.cluster_centers_da.sel(cluster=label_).plot(**kw)
            title = '{:4.1f}%'.format(self.get_cluster_fraction(label_)*100)
            plt.title(title)
