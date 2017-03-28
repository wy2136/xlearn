'''
Author: Wenchang Yang (yang.wenchang@uci.edu)
'''
import xarray as xr
import numpy as np
from sklearn.decomposition import PCA as PCA_sklearn


class PCA(PCA_sklearn):
    '''Inherit from sklearn.linear_model.LinearRegression that accept xarray.DataArray data
    to fit.'''

    def __init__(self, *args, **kw):
        weight = kw.pop('weight', None)
        n_components = kw.pop('n_components', 1)
        super().__init__(*args, n_components=n_components, **kw)
        self.weight = weight
    def __str__(self):
        attrs = ['copy', 'iterated_power', 'n_components', 'random_state', 'svd_solver',
            'tol', 'whiten']
        # s = ', '.join(['{}={}'.format(a, getattr(self, a)) for a in attrs])
        values = []
        for attr in attrs:
            value = getattr(self, attr)
            if isinstance(value, str):
                value = '"{}"'.format(value)
            values.append(value)
        s = ', '.join(['{}={}'.format(attr, value)
            for attr,value in zip(attrs, values)])
        return '[PCA for xarray]: {}.'.format(s)
    def __repr__(self):
        return self.__str__()

    def fit(self, da):
        '''xarray.DataArray version of sklearn.decomposition.PCA.fit.'''

        # Compatible with the slearn.decomposition.PCA fit method when the input data are not DataArray
        da = da.copy()
        
        if not isinstance(da, xr.DataArray):
            X = da
            return super().fit(X)
        
        if self.weight is not None:
            da *= self.weight
            
        # input data parameters
        X = da.data
        n_samples = X.shape[0]
        grid_shape = X.shape[1:]
        n_grids = np.prod(grid_shape)
        X = X.reshape((n_samples, n_grids))
        valid_grids = ~np.isnan(X[0, :])
        X = X[:, valid_grids]
        
        # call the sklearn version model
        super().fit(X)
        
        # reshape the coefficient array back
        eofs = np.empty((self.n_components_, n_grids)) * np.nan
        eofs[:, valid_grids] = self.components_
        eofs = eofs.reshape((self.n_components_,) + grid_shape)
        # reshape the mean_
        mean_ = np.empty(n_grids) * np.nan
        mean_[valid_grids] = self.mean_
        mean_ = mean_.reshape(grid_shape)
        
        # wrap regression coefficient into DataArray
        # dims
        grid_dims = da.dims[1:]
        eofs_dims = ('mode',) + grid_dims
        # coords
        grid_coords = {dim: da[dim] for dim in grid_dims}
        eofs_coords = grid_coords.copy()
        eofs_coords[eofs_dims[0]] = np.arange(self.n_components_)
        # DataArray
        self.components_da = xr.DataArray(eofs,
            dims=eofs_dims, coords=eofs_coords)
        # self.mean_
        self.mean_da = xr.DataArray(mean_,
            dims=grid_dims, coords=grid_coords)
        
        
        return self
    
    def transform(self, da):
        '''xarray version of sklearn.decomposition.PCA.transform'''
        # Compatible with the slearn.decomposition.PCA fit method when the input data are not DataArray
        da = da.copy()
        if not isinstance(da, xr.DataArray):
            X = da
            return super().transform(X)
        
        if self.weight is not None:
            da *= self.weight
            
        # input data parameters
        X = da.data
        n_samples = X.shape[0]
        grid_shape = X.shape[1:]
        n_grids = np.prod(grid_shape)
        X = X.reshape((n_samples, n_grids))
        valid_grids = ~np.isnan(X[0, :])
        X = X[:, valid_grids]
        
        # call the sklearn version model
        pcs = super().transform(X)
        
        # wrap regression coefficient into DataArray
        # dims
        sample_dim = da.dims[0]
        pcs_dims = (sample_dim, 'mode')
        # coords
        pcs_coords = {sample_dim: da[sample_dim],
                        'mode': np.arange(self.n_components_)}
        # DataArray
        pcs_da = xr.DataArray(pcs,
            dims=pcs_dims, coords=pcs_coords)
        
        return pcs_da
        
    def inverse_transform(self, da):
        '''xarray version of sklearn.decomposition.PCA.inverse_transform'''
        # Compatible with the slearn.decomposition.PCA fit method when the input data are not DataArray
        da = da.copy()
        if not isinstance(da, xr.DataArray):
            X = da
            return super().inverse_transform(X)
        
        # parameters
        pcs = da.data
        n_samples = pcs.shape[0]
        eofs_da = self.components_da
        grid_shape = eofs_da.shape[1:]
        n_grids = np.prod(grid_shape)
        valid_grids = ~np.isnan( eofs_da.sel(mode=0).data.reshape((n_grids,)) )
        
        
        # call the sklearn version model
        X = np.empty((n_samples, n_grids)) * np.nan
        X[:, valid_grids] = super().inverse_transform(pcs)
        X = X.reshape((n_samples,) + grid_shape )
        
        # wrap into DataArray
        X_dims = (da.dims[0],) + eofs_da.dims[1:]
        X_coords = {da.dims[0]: da[da.dims[0]]}
        for dim in eofs_da.dims[1:]:
            X_coords[dim] = eofs_da[dim]
        X_da = xr.DataArray(X, dims=X_dims, coords=X_coords)
        
        if self.weight is not None:
            X_da /= self.weight
        
        return X_da
        
        