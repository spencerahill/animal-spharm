import numpy as np
import pytest
import xarray as xr

from aospy_user import SpharmInterface


@pytest.fixture
def compute_vrtdiv(u, v):
    sphint = SpharmInterface(u, v)
    sphint.make_vectorwind()
    sphint.make_spharmt()

    vort, divg = sphint.vectorwind.vrtdiv()
    return sphint.to_xarray(vort), sphint.to_xarray(divg)


def test_vrtdiv():
    path = ('/archive/Spencer.Hill/am2/am2clim_reyoi/gfdl.ncrc2-default-prod/'
            'pp/atmos_level/ts/monthly/1yr/atmos_level.198301-198312.')

    # Vertically defined, sigma levels.
    u_arr = xr.open_dataset(path + 'ucomp.nc').ucomp
    v_arr = xr.open_dataset(path + 'vcomp.nc').vcomp
    vort, divg = compute_vrtdiv(u_arr, v_arr)
    assert vort.shape == u_arr.shape
    assert divg.shape == u_arr.shape
    np.testing.assert_array_equal(u_arr.lat, vort.lat)
    np.testing.assert_array_equal(u_arr.lon, vort.lon)
    np.testing.assert_array_equal(u_arr.time, vort.time)
    np.testing.assert_array_equal(u_arr.pfull, vort.pfull)

    # Not vertically defined.
    u0 = u_arr[:,0]
    v0 = v_arr[:,0]
    vort0, divg0 = compute_vrtdiv(u0, v0)
    assert vort0.shape == u0.shape
    assert divg0.shape == u0.shape

    # Dummy case: zeros everywhere
    u_arr_zeros = xr.DataArray(np.zeros_like(u_arr.values), dims=u_arr.dims,
                               coords=u_arr.coords)
    v_arr_zeros = u_arr_zeros.copy()
    vort_zeros, divg_zeros = compute_vrtdiv(u_arr_zeros, v_arr_zeros)
    assert not vort_zeros.any()
    assert not divg_zeros.any()
