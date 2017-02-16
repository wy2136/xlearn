'''
Author: Wenchang Yang (yang.wenchang@uci.edu)
'''
import xarray as xr
import numpy as np
from sklearn.linear_model import LinearRegression as LinearRegression_sklearn
from scipy import stats


class LinearRegression(LinearRegression_sklearn):
    '''Inherit from sklearn.linear_model.LinearRegression that accept xarray.DataArray data
    to fit.'''

    def __init__(self, *args, **kw):
        n_jobs = kw.pop('n_jobs', -1)
        normalize_x = kw.pop('normalize', False)
        normalize_x = kw.pop('normalize_x', normalize_x)
        normalize_y = kw.pop('normalize_y', False)
        normalize_xy = kw.pop('normalize_xy', False)
        super().__init__(*args, n_jobs=n_jobs, **kw)
        if normalize_xy:
            normalize_x = normalize_xy
            normalize_y = normalize_xy
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
    def __str__(self):
        attrs = ['copy_X', 'fit_intercept', 'n_jobs', 'normalize_x', 'normalize_y']
        # s = ', '.join(['{}={}'.format(a, getattr(self, a)) for a in attrs])
        values = []
        for attr in attrs:
            value = getattr(self, attr)
            if isinstance(value, str):
                value = '"{}"'.format(value)
            values.append(value)
        s = ', '.join(['{}={}'.format(attr, value)
            for attr,value in zip(attrs, values)])
        return '[LinearRegression for xarray]: {}.'.format(s)
    def __repr__(self):
        return self.__str__()

    def fit(self, da_X, da_Y):
        '''xarray.DataArray version of sklearn.linear_model.LinearRegression.fit.'''

        # Compatible with the slearn.linear_model.LinearRegression fit method when the input data are not DataArray
        if not isinstance(da_X, xr.DataArray) \
            or not isinstance(da_Y, xr.DataArray):
            X = np.array(da_X)
            if X.ndim == 1:
                X = X[:, np.newaxis]
            if self.normalize_x:
                X = ( X - X.mean(axis=0, keepdims=True))/X.std(axis=0, keepdims=True)
            
            Y = np.array(da_Y)
            if self.normalize_y:
                Y = ( Y - Y.mean(axis=0, keepdims=True))/Y.std(axis=0, keepdims=True)
            
            return super().fit(X, Y)

        # input data parameters
        X = da_X.data
        if X.ndim == 1:
            da_X_is_1d = True
            X = X[:, np.newaxis]
        else:
            da_X_is_1d = False
        if self.normalize_x:
            X = ( X - X.mean(axis=0, keepdims=True))/X.std(axis=0, keepdims=True)
        Y = da_Y.data
        if Y.ndim == 1:
            da_Y_is_1d = True
            Y = Y[:, np.newaxis]
        else:
            da_Y_is_1d = False
        n_samples = X.shape[0]
        n_features = X.shape[1]
        grid_shape = Y.shape[1:]
        n_grids = np.prod(grid_shape)

        # reshape da_Y from n-d to 2-d
        Y = Y.reshape((n_samples, n_grids))
        valid_grids = ~np.isnan(Y[0, :])
        Y = Y[:, valid_grids]
        if self.normalize_y:
            Y = ( Y - Y.mean(axis=0, keepdims=True))/Y.std(axis=0, keepdims=True)
        
        
        # call the sklearn version model
        super().fit(X, Y)
        
        # reshape the coefficient array back
        b = np.empty((n_grids, n_features)) * np.nan
        b[valid_grids, :] = self.coef_
        b = b.reshape(grid_shape + (n_features,))
        
        # wrap regression coefficient into DataArray
        if da_Y_is_1d:
            grid_dims = ('grid',)
        else:
            grid_dims = da_Y.dims[1:]
        if da_X_is_1d:
            feature_dim = 'feature'
        else:
            feature_dim = da_X.dims[1]
        coef_dims = grid_dims + (feature_dim,)
        
        if da_X_is_1d:
            coef_coords = {feature_dim: np.array([0])}
        else:
            coef_coords = {feature_dim: da_X[feature_dim]}
        if da_Y_is_1d: 
            coef_coords[grid_dims[0]] = np.array([0])
        else:
            for dim in grid_dims:
                coef_coords[dim] = da_Y[dim]
                
        self.coef_da = xr.DataArray(b, dims=coef_dims, coords=coef_coords)
        if da_X_is_1d:
            self.coef_da = self.coef_da.sel(feature=0)
        if da_Y_is_1d:
            self.coef_da = self.coef_da.sel(grid=0)
        
        # add-on estimate: p_value
        p = np.empty( (n_grids, n_features) ) * np.nan
        sse = np.sum( (self.predict(X) - Y)**2, axis=0 )[:, np.newaxis]
        dof = n_samples - n_features - 1
        Xa = X - X.mean(axis=0, keepdims=True)
        ssx_inv = np.diagonal( np.linalg.inv( np.dot(Xa.T, Xa) ) )[np.newaxis,:]
        se = np.sqrt( sse * ssx_inv / dof )
        t = self.coef_ / se
        p[valid_grids, :] = 2 * (1 - stats.t.cdf(np.abs(t), dof))
        p = p.reshape( grid_shape + (n_features,) )
        self.pvalue_da = xr.DataArray(p, dims=coef_dims, coords=coef_coords)
        if da_X_is_1d:
            self.pvalue_da = self.pvalue_da.sel(feature=0)
        if da_Y_is_1d:
            self.pvalue_da = self.pvalue_da.sel(grid=0)
        
        # add-on: intercept
        b0 = np.empty((n_grids,)) * np.nan
        b0[valid_grids] = self.intercept_
        b0 = b0.reshape(grid_shape)
        intercept_dims = grid_dims
        intercept_coords = {}
        if da_Y_is_1d:
            intercept_coords[grid_dims[0]] = np.array([0])
        else:
            for dim in grid_dims:
                intercept_coords[dim] = da_Y[dim]
            
        self.intercept_da = xr.DataArray(b0, dims=intercept_dims,
            coords=intercept_coords)
        if da_Y_is_1d:
            self.intercept_da = self.intercept_da.sel(grid=0)
        
        return self