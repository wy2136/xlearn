'''
Maching learning accessor for xarray.

Author: Wenchang Yang (yang.wenchang@uci.edu)
'''

import xarray as xr
from . import cluster, linear_model, decomposition

@xr.register_dataarray_accessor('learn')
class LearnAccessor(object):
    def __init__(self, da):
        self._obj = da

    def kmeans(self, *args, **kw):
        '''See xlearn.cluster.Kmeans.fit for description.'''
        return cluster.KMeans(*args, **kw).fit(self._obj)

    def regress(self, da_x, *args, **kw):
        '''See xlearn.linear_model.LinearRegression.fit for description'''
        return linear_model.LinearRegression(*args, **kw).fit(da_x, self._obj)

    def pca(self, *args, **kw):
        '''See xlearn.decomposition.PCA.fit for description'''
        return decomposition.PCA(*args, **kw).fit(self._obj)
