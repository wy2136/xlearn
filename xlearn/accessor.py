'''
Maching learning accessor for xarray.

Author: Wenchang Yang (yang.wenchang@uci.edu)
'''

import xarray as xr
from . import cluster

@xr.register_dataarray_accessor('learn')
class LearnAccessor(object):
    def __init__(self, da):
        self._obj = da

    def kmeans(self, *args, **kw):
        '''See xlearn.cluster.Kmeans.fit for description.'''
        return cluster.KMeans(*args, **kw).fit(self._obj)
