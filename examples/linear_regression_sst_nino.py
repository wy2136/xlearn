from xlearn.linear_model import LinearRegression
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pyingrid import Ingrid

sst = Ingrid('ersst4').do('T(Jan 1980)(Dec 2015)RANGE')\
    .do('T(Dec)VALUES').to_xarray().load()

nino12 = sst.sel(X=slice(270,280), Y=slice(-10,0)).geo.fldmean()
nino3 = sst.sel(X=slice(210,270), Y=slice(-5,5)).geo.fldmean()
nino34 = sst.sel(X=slice(190,240), Y=slice(-5,5)).geo.fldmean()
nino4 = sst.sel(X=slice(160,210), Y=slice(-5,5)).geo.fldmean()

plt.figure()
r12 = sst.learn.regress(nino12, normalize_x=True)
r12.coef_da.geo.plot(fill_continents=True, units='K')
plt.title('SST Regressed on Normalized Nino1.2')
r12.pvalue_da.where(r12.pvalue_da<0.05).geo.plot(fill_continents=True,
    plot_type='hatch')
# plt.tight_layout(); plt.savefig('fig-regress-sst-nino12.pdf')

plt.figure()
r3 = sst.learn.regress(nino3, normalize_x=True)
r3.coef_da.geo.plot(fill_continents=True, units='K')
plt.title('SST Regressed on Normalized Nino3')
r3.pvalue_da.where(r3.pvalue_da<0.05).geo.plot(fill_continents=True,
    plot_type='hatch')
# plt.tight_layout(); plt.savefig('fig-regress-sst-nino3.pdf')

plt.figure()
r34 = sst.learn.regress(nino34, normalize_x=True)
r34.coef_da.geo.plot(fill_continents=True, units='K')
plt.title('SST Regressed on Normalized Nino3.4')
r34.pvalue_da.where(r34.pvalue_da<0.05).geo.plot(fill_continents=True,
    plot_type='hatch')
# plt.tight_layout(); plt.savefig('fig-regress-sst-nino34.pdf')

plt.figure()
r4 = sst.learn.regress(nino4, normalize_x=True)
r4.coef_da.geo.plot(fill_continents=True, units='K')
plt.title('SST Regressed on Normalized Nino4')
r4.pvalue_da.where(r4.pvalue_da<0.05).geo.plot(fill_continents=True,
    plot_type='hatch')
# plt.tight_layout(); plt.savefig('fig-regress-sst-nino4.pdf')

    