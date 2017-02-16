from xlearn.linear_model import LinearRegression
import xarray as xr
import numpy as np

import statsmodels.api as sm

# y = b0 + b1*x + e
np.random.seed(0)
x = np.random.randn(100).reshape(100,1)
np.random.seed(1)
e = np.random.randn((100)).reshape(100, 1)
y = 2*x + e*4

da_X = xr.DataArray(x, dims=['time', 'feature'])
da_Y = xr.DataArray(y, dims=['time', 'grid'])

m = LinearRegression()
m.fit(da_X, da_Y)
print('[xlearn results]:', '-'*10)
print('[coefficient]')
print(m.coef_da)
print('[pvalues]')
print(m.p_da)

# compared with statsmodel
r = sm.OLS(y, sm.add_constant(x)).fit()
print('\n[statsmodels results]:', '-'*10)
print('[coefficient]')
print(r.params)
print('[pvalues]')
print(r.pvalues)

sys.exit()

# y = b0 + b1*x1 + b2*x2 + e
np.random.seed(0)
xx = np.random.randn(200).reshape(100,2)
np.random.seed(1)
e = np.random.randn((100)).reshape(100, 1)
y = xx[:,0:1] + xx[:, 1:2]*2 + e*4

da_X = xr.DataArray(xx, dims=['time', 'feature'])
da_Y = xr.DataArray(y, dims=['time', 'grid'])

m = LinearRegression()
m.fit(da_X, da_Y)
print('[xlearn results]:', '-'*10)
print('[coefficient]')
print(m.coef_da)
print('[pvalues]')
print(m.p_da)

# compared with statsmodel
r = sm.OLS(y, sm.add_constant(xx)).fit()
print('\n[statsmodels results]:', '-'*10)
print('[coefficient]')
print(r.params)
print('[pvalues]')
print(r.pvalues)


# y_{ij} = b0 + b1*x1 + b2*x2 + e
np.random.seed(0)
xx = np.random.randn(200).reshape(100,2)
np.random.seed(1)
e = np.random.randn((100)).reshape(100, 1)
y = xx[:,0:1] + xx[:, 1:2]*2 + e*4
yy = np.empty((100, 3, 4)) * np.nan
for i in range(4):
    yy[:, 0, i:i+1] = y * (i+1)
    yy[:, 2, i:i+1] = y * (i+1)

da_X = xr.DataArray(xx, dims=['time', 'feature'])
da_Y = xr.DataArray(yy, dims=['time', 'lat', 'lon'])

m = LinearRegression()
m.fit(da_X, da_Y)
print('[xlearn results]:', '-'*10)
print('[coefficient]')
print(m.coef_da)
print('[pvalues]')
print(m.p_da)
    
