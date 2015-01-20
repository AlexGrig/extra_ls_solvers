# -*- coding: utf-8 -*-
"""
author: Alexander Grigorevskiy, 2013

Module provides some extra functions to solve least-squares problems.
For those extra functions to work the special recompilation of SciPy is 
currently required. The example of the file which needs to be changed in SciPy
is provided in the same folder (flapack_user.pyf.src.example). This file is
contained in scipy/linalg. Merging may be required for subsequent versions of 
scipy.

If the SciPy is not recompiled then functions resort to the original SciPy
functions.

Currently the fastest method is ls_cof. This should be prefered, however the
ls_svdb is very close. 
"""


import numpy as np

import scipy.linalg as la
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.misc import LinAlgError, _datacopied




def ls_scipy_current_solver(a, b, cond=None, overwrite_a=False, overwrite_b=False,
          check_finite=True):
    """
    This function just calls the lstsq (least squares solver from the scipy)
    """

    return la.lstsq( a, b, cond, overwrite_a, overwrite_b, check_finite)

def ls_numpy_standard_solver(a, b, cond=-1):
    """
    This function just calls the lstsq (least squares solver from the scipy)
    """

    return np.linalg.lstsq( a, b, cond)


def ls_cof(a, b, cond=None, overwrite_a=False, overwrite_b=False,
          check_finite=True):

    """
    Compute minimum norm least-squares solution to the equation Ax = b.
    The fastest algorithm.

    Accepts multiple right-hand sides.

    Uses complete orthogonal factorization of A.

    Input:
        cond : float, optional. Is used to determine the effective rank of A, which
               is defined as the order of the largest leading triangular
               submatrix R11 in the QR factorization with pivoting of A,
               whose estimated condition number < 1/RCOND.
        overwrite_a : bool, optional
                    Discard data in `a` (may enhance performance). Default is False.
        overwrite_b : bool, optional
                    Discard data in `b` (may enhance performance). Default is False.
        check_finite : boolean, optional
                   Whether to check the input matrixes contain only finite numbers.
                   Disabling may give a performance gain, but may result to problems
                   (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Output:
        x : solution
        resids : residuals
        rank : determined numerical rank of a
        j : some additional info
    """

    if check_finite:
        a1,b1 = map(np.asarray_chkfinite, (a,b))
    else:
        a1,b1 = map(np.asarray, (a,b))
    if len(a1.shape) != 2:
        raise ValueError('expected matrix')
    m, n = a1.shape
    if len(b1.shape) == 2:
        nrhs = b1.shape[1]
    else:
        nrhs = 1
    if m != b1.shape[0]:
        raise ValueError('incompatible dimensions')

    try:
        gelsy, = get_lapack_funcs(('gelsy',), (a1, b1))
    except ValueError:
        # If the LAPACK gelsy function is not found resort to the standart implementation
        return la.lstsq( a, b, cond, overwrite_a, overwrite_b, check_finite)

    if n > m:
        # need to extend b matrix as it will be filled with
        # a larger solution matrix
        if len(b1.shape) == 2:
            b2 = np.zeros((n, nrhs), dtype=gelsy.dtype)
            b2[:m,:] = b1
        else:
            b2 = np.zeros(n, dtype=gelsy.dtype)
            b2[:m] = b1
        b1 = b2

    overwrite_a = overwrite_a or _datacopied(a1, a)
    overwrite_b = overwrite_b or _datacopied(b1, b)

    jptv = np.zeros( (n,1), dtype=np.int32 )

    work = gelsy(a1, b1, jptv, lwork=-1)[4]
    lwork = work[0].real.astype(np.int)
    v, x, j, rank, work, info = gelsy(
        a1, b1, jptv, cond=cond, lwork=lwork, overwrite_a=overwrite_a,
        overwrite_b=overwrite_b)
    # v - working matrix - A
    # x - solution
    # j - jptv - some add info
    # rank - determined numerical rank
    # work - some working array
    # info - error indocation

    if info > 0:
        raise LinAlgError("Unusual behavior. This should not happen.")
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal gelsy'
                                                                    % -info)
    resids = np.asarray([], dtype=x.dtype)
    if n < m:
        x1 = x[:n]
        if rank == n:
            resids = np.sum(np.abs(x[n:])**2, axis=0)
        x = x1
    return x, resids, rank, j

def ls_svdb(a, b, cond=None, overwrite_a=False, overwrite_b=False,
          check_finite=True):
    """
    Compute minimum norm least-squares solution to the equation Ax = b.
    Very close to fasterst but still second.

    Accepts multiple right-hand sides.

    Uses block SVD algorithm ( divide-and-conquer SVD) which is faster then regular SVD if the
    matrix is large.

    Input:
        a : (M, N) array_like
            Left hand side matrix (2-D array).
        b : (M,) or (M, K) array_like
            Right hand side matrix or vector (1-D or 2-D array).
        cond : float, optional
            Cutoff for 'small' singular values; used to determine effective
            rank of a. Singular values smaller than
            ``rcond * largest_singular_value`` are considered zero.
        overwrite_a : bool, optional
            Discard data in `a` (may enhance performance). Default is False.
        overwrite_b : bool, optional
            Discard data in `b` (may enhance performance). Default is False.
        check_finite : boolean, optional
            Whether to check the input matrixes contain only finite numbers.
            Disabling may give a performance gain, but may result to problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Output:
        x : solution
        resids : residuals
        rank: effective rank of the matrix a
        s : singular values
    """

    if check_finite:
        a1,b1 = map(np.asarray_chkfinite, (a,b))
    else:
        a1,b1 = map(np.asarray, (a,b))
    if len(a1.shape) != 2:
        raise ValueError('expected matrix')
    m, n = a1.shape
    if len(b1.shape) == 2:
        nrhs = b1.shape[1]
    else:
        nrhs = 1
    if m != b1.shape[0]:
        raise ValueError('incompatible dimensions')
    try:
        gelsd, = get_lapack_funcs(('gelsd',), (a1, b1))
    except:
         # If the LAPACK gelsd function is not found resort to the standart implementation
        return la.lstsq( a, b, cond, overwrite_a, overwrite_b, check_finite)

    if n > m:
        # need to extend b matrix as it will be filled with
        # a larger solution matrix
        if len(b1.shape) == 2:
            b2 = np.zeros((n, nrhs), dtype=gelsd.dtype)
            b2[:m,:] = b1
        else:
            b2 = np.zeros(n, dtype=gelsd.dtype)
            b2[:m] = b1
        b1 = b2

    overwrite_a = overwrite_a or _datacopied(a1, a)
    overwrite_b = overwrite_b or _datacopied(b1, b)

    v, x, s, rank, work, iwork, info = gelsd(a1, b1, 1, lwork=-1)
    lwork = work[0].real.astype(np.int)
    iwork_size = iwork[0].real.astype(np.int)


    v, x, s, rank, work, iwork, info = gelsd(
        a1, b1, iwork_size, cond=cond, lwork=lwork, overwrite_a=overwrite_a,
        overwrite_b=overwrite_b)

    # v - working matrix - A
    # x - solution
    # s - singular values
    # rank - determined numerical rank
    # work, iwork - some working arrays
    # info - error indocation

    if info > 0:
        raise LinAlgError("SVD did not converge in Linear Least Squares")
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal gelsd'
                                                                    % -info)
    resids = np.asarray([], dtype=x.dtype)
    if n < m:
        x1 = x[:n]
        if rank == n:
            resids = np.sum(np.abs(x[n:])**2, axis=0)
        x = x1
    return x, resids, rank, s

if __name__ == '__main__':
    pass



