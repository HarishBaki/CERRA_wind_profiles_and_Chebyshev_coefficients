
# === importing dependencies ===#
import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
import ast
import yaml
import time

# we would like to make sure that the reference height levels cover the profiler levels and NOW23 levels, including the 0 m level.
profilers_levels = np.array([10] + list(range(100, 501, 25)))
NOW23_levels = np.array([10] + list(range(20, 301, 20)) + [400, 500])
CERRA_levels = np.array([10,15, 30, 50., 75., 100., 150., 200., 250., 300., 400., 500])
# combine the two arrays and remove duplicates
ref_H = np.unique(np.concatenate((np.array([0]),profilers_levels, NOW23_levels, CERRA_levels)))

# These variables are used while creating the Chebyshev coefficients
poly_order = 4
CPtype = 1

def normalize(H,ref_H=ref_H):
    '''
    Normalizes the height levels between -1 and 1
    ref_H: A vector of reference height levels, in our case CERRA levels
    H: A vector of height levels
    '''
    a = 2 / (np.max(ref_H) - np.min(ref_H))
    b = - (np.max(ref_H) + np.min(ref_H)) / (np.max(ref_H) - np.min(ref_H))
    Hn = a * ref_H + b

    Hn = np.interp(H, ref_H, Hn)
    return Hn

def Chebyshev_Basu(x, poly_order=poly_order, CPtype=CPtype):
    '''
    This function computes the Chebyshev polynomials, according to the equations in Mason, J. C., & Handscomb, D. C. (2002). Chebyshev polynomials. Chapman and Hall/CRC.
    x: input variable, between -1 and 1
    poly_order: order of polynomials
    CPtype: 1 or 2, according to the publication
    '''
    if x.ndim == 1:
        x = x[:, np.newaxis]

    CP = np.zeros((len(x), poly_order + 1))

    CP[:, 0] = 1  # T0(x) = 1
    if poly_order >= 1:
        if CPtype == 1:  # Chebyshev polynomial of first kind
            CP[:, 1] = x.flatten()  # T1(x) = x
        else:  # Chebyshev polynomial of second kind
            CP[:, 1] = 2 * x.flatten()  # T1(x) = 2x
        if poly_order >= 2:
            for n in range(2, poly_order + 1):
                CP[:, n] = 2 * x.flatten() * CP[:, n - 1] - CP[:, n - 2]
    return CP

def Chebyshev_Coeff(H, U,poly_order=poly_order,CPtype=CPtype,ref_H=ref_H):
    '''
    This function computes the Chebyshev coefficients through inverse transform of system of linear equations
    H: height levels, in their useual units
    U: wind speed at the height levels
    p: polynomial order
    CPtype: 1 or 2, according to the publication
    '''
    H = H.flatten()
    U = U.flatten()

    # Normalize H
    Hn = normalize(H, ref_H=ref_H)
    
    # Remove NaN values
    Indx = np.where(~np.isnan(U))[0]
    Ha = Hn[Indx]
    Ua = U[Indx]
    N = len(Ua)

    # Linearly extrapolate wind values at the boundaries
    #spline_left = interp1d(Ha, Ua, kind='linear', fill_value='extrapolate')
    #Uax = spline_left([-1])

    #spline_right = interp1d(Ha, Ua, kind='linear', fill_value='extrapolate')
    #Uay = spline_right([1])
    #Ua = np.concatenate([Uax, Ua, Uay])
    #Ha = np.concatenate([-1 + np.zeros(1), Ha, 1 + np.zeros(1)])       # these two seems are unnecessary, which bring adidtional offset due to extrapolation.
    
    # Predict the gap-filled and denoised profile
    PL = Chebyshev_Basu(Ha, poly_order=poly_order, CPtype=CPtype)
    # Compute the coefficients C
    Coeff = np.linalg.pinv(PL) @ Ua
    return Coeff

def WindProfile(Z,Coeff, poly_order=poly_order, CPtype=CPtype,ref_H=ref_H):
    '''
    This function computes the full level wind profile provided vertical levels and the Chebyshev coefficients
    Z: height levels, in their useual units
    Coeff: Chebysev coefficients
    '''
    # Normalize H
    Hn = normalize(Z, ref_H=ref_H)
    PL_full = Chebyshev_Basu(Hn, poly_order=poly_order, CPtype=CPtype)
    Mp = PL_full @ Coeff
    return Mp

def Basu_WindProfile(z, a,b,c):
    '''
    This function computes the full level wind profile provided vertical levels and the Basu coefficients
    Z: height levels, in their useual units
    a,b,c: Basu coefficients
    '''
    return 1 / (a + b * np.log(z) + c * (np.log(z))**2)

def chebyshev(y: np.ndarray) -> np.ndarray:
    """Calculate Chebyshev coefficients for input y"""
    return Chebyshev_Coeff(CERRA_levels, y)


def chebyshev_vec(y: xr.DataArray, dim: str) -> xr.DataArray:
    """
    Calculate Chebyshev coefficients for input y in `dim` direction.
    Sample usage cheb = chebyshev_vec(ds, dim="heightAboveGround")
    """
    cheb = xr.apply_ufunc(
        lambda y_i: xr.DataArray(chebyshev(y_i), dims=["coeff"]),  # we need to make the output a DataArray with new dim
        y,  # entire dataset as input
        input_core_dims=[[dim]],  # dimension along which to apply chebyshev
        output_core_dims=[["coeff"]],  # dimension of the output, which is the coefficients
        vectorize=True,  # vectorize the operation
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"coeff": poly_order+1}},
    )
    cheb.name = "Chebyshev_coefficients"
    return cheb