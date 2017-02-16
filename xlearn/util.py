'''
Author: Wenchang Yang (yang.wenchang@uci.edu)
'''
import numpy as np

def _reshape_data(X):
    """Reshape a ndarray X into a 2d array that can be used in sklearn models.
    
    The first dimension of X is usually 'time' (serves as the 'sample' dimension
    in the sklearn dataset). 
    
    The remaining dimensions are usually spatial grids,
    e.g. longitude, latitude or level. 
    
    A typical X has a shape of (n_time, n_lat, n_lon).
    
    # returns:
        X_: reshaped X with shape (n_samples, n_grids)
        params: dict with keys: n_samples, grid_shape, n_grids, and valid_grids.
    """
    n_samples = X.shape[0]
    grid_shape = X.shape[1:]
    n_grids = np.prod(grid_shape)
    
    X_ = X.reshape((n_samples, n_grids))
    valid_grids = ~np.isnan(X_[0,:])
    X_ = X_[:, valid_grids]
    
    params = dict(n_samples=n_samples, grid_shape=grid_shape, 
        n_grids=n_grids, valid_grids=valid_grids)
    
    return X_, params