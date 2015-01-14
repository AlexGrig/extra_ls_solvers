# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:43:49 2013

@author: agrigori
"""
import numpy as np

from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.misc import LinAlgError, _datacopied

def ls_svd(a, b, cond=None, overwrite_a=False, overwrite_b=False,
          check_finite=True):
    """
    Don not use this function, use lstsq from scipy!!!
	
    Original scipy function. Accepts multiple right-hand sides.    
    The most slowest algorithm    
    
    Compute least-squares solution to equation Ax = b.

    Compute a vector x such that the 2-norm ``|b - A x|`` is minimized.

    Parameters
    ----------
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

    Returns
    -------
    x : (N,) or (N, K) ndarray
        Least-squares solution.  Return shape matches shape of `b`.
    residues : () or (1,) or (K,) ndarray
        Sums of residues, squared 2-norm for each column in ``b - a x``.
        If rank of matrix a is < N or > M this is an empty array.
        If b was 1-D, this is an (1,) shape array, otherwise the shape is (K,).
    rank : int
        Effective rank of matrix `a`.
    s : (min(M,N),) ndarray
        Singular values of `a`. The condition number of a is
        ``abs(s[0]/s[-1])``.

    Raises
    ------
    LinAlgError :
        If computation does not converge.


    See Also
    --------
    optimize.nnls : linear least squares with non-negativity constraint

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
    gelss, = get_lapack_funcs(('gelss',), (a1, b1))
    if n > m:
        # need to extend b matrix as it will be filled with
        # a larger solution matrix
        if len(b1.shape) == 2:
            b2 = np.zeros((n, nrhs), dtype=gelss.dtype)
            b2[:m,:] = b1
        else:
            b2 = np.zeros(n, dtype=gelss.dtype)
            b2[:m] = b1
        b1 = b2

    overwrite_a = overwrite_a or _datacopied(a1, a)
    overwrite_b = overwrite_b or _datacopied(b1, b)

    # get optimal work array
    work = gelss(a1, b1, lwork=-1)[4]
    lwork = work[0].real.astype(np.int)
    v, x, s, rank, work, info = gelss(
        a1, b1, cond=cond, lwork=lwork, overwrite_a=overwrite_a,
        overwrite_b=overwrite_b)

    if info > 0:
        raise LinAlgError("SVD did not converge in Linear Least Squares")
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal gelss'
                                                                    % -info)
    resids = np.asarray([], dtype=x.dtype)
    if n < m:
        x1 = x[:n]
        if rank == n:
            resids = np.sum(np.abs(x[n:])**2, axis=0)
        x = x1
    return x, resids, rank, s


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
    gelsy, = get_lapack_funcs(('gelsy',), (a1, b1))
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
    gelsd, = get_lapack_funcs(('gelsd',), (a1, b1))
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



def test1():
    """
         Test the speed of three ls solvers on not full rank matrices.
         Also keep track of difference in the solutions.
    """
    import scipy as sp
    import math
    from numpy import random as rnd    
    import time
    
    times_1 = []
    times_2 = []    
    times_3 = []
    diff1 = []    
    diff2 = []
    diff3 = []
    
    m_steps = range(100,1000,100 ) + range(1000, 11000, 1000)
    #m_steps = range(100,300,100)
    for (i,m) in enumerate( m_steps ):

        n  = math.ceil(2./3. * m )
        k = math.ceil(1./2. * m )        
        
        A =  10 * rnd.rand(m,k) - 5
        Temp =  10 * rnd.rand(k,n) - 5       
        
        A = A.dot(Temp)
        
        b =  10 * rnd.rand(m,1) - 5        
        
        A = np.asfortranarray(A)
        
        A1 = A.copy(); b1 = b.copy()             
        t1 = time.time()
        res1 = ls_svd(A1,b1,cond=1e-6,overwrite_a=False, overwrite_b=False)
        t1 = time.time() - t1          
                           
        A2 = A.copy(); b2 = b.copy()                  
        t2 = time.time()
        res2 = ls_cof(A2,b2,cond=1e-6,overwrite_a=False, overwrite_b=False)
        t2 = time.time() - t2
        
        d1 = np.max( res1[0] - res2[0] )        
        
        
        A3 = A.copy(); b3 = b.copy()             
        t3 = time.time()
        res3 = ls_svdb(A3,b3,cond=1e-6,overwrite_a=False, overwrite_b=False)
        t3 = time.time() - t3
        
        d2 = np.max( res1[0] - res3[0] )        
        d3 = np.max( res2[0] - res3[0] )
        
        times_1.append(t1)
        times_2.append(t2)
        times_3.append(t3)
        diff1.append(d1)
        diff2.append(d2)
        diff3.append(d3)
        
    import matplotlib.pyplot as plt        
    
    plt.title('Algorihms speed comparison', size = 18, weight = 'bold')        
    plt.plot(m_steps, times_1, 'ro', label = 'SVD')        
    plt.plot(m_steps, times_2, 'bo', label = 'COF')              
    plt.plot(m_steps, times_3, 'go', label = 'Block SVD')      
        
    plt.ylabel('Seconds', size = 15, weight = 'heavy' )
    plt.xlabel('Matrix m-size, n-size = 2/3 m-size, rank = 1/2 m-size',\
                size = 15, weight = 'bold' )
    
    plt.legend(loc=2)
    plt.show()
    
    # Saved to ls_test_speed_comparison1.png 
    # and ls_test_accuracy_comparison1.png   
    
    
    return m_steps, times_1, times_2, times_3, diff1, diff2, diff3

def test2():
    """
    Test how ls scales with respect to dimesion m, under fixed dimension
    n and rank k.
    
    Test only fastest algorithms COF and bSVD
    """
          
    import scipy as sp
    import math
    from numpy import random as rnd    
    import time
    
    times_2 = []    
    times_3 = []
    diff3 = []

    n = 2000 # Fixed second dimension          
    k = 1000 # Fixed rank  
    
    m_steps = range(100,1000,100 ) + range(1000, 16000, 1000)
    #m_steps = range(100,300,100)
        
    for (i,m) in enumerate( m_steps ):
        
        A =  10 * rnd.rand(m,k) - 5
        Temp =  10 * rnd.rand(k,n) - 5       
        
        A = A.dot(Temp)
        
        b =  10 * rnd.rand(m,1) - 5        
        
        A = np.asfortranarray(A)
        
                           
        A2 = A.copy(); b2 = b.copy()                  
        t2 = time.time()
        res2 = ls_cof(A2,b2,cond=1e-6,overwrite_a=False, overwrite_b=False)
        t2 = time.time() - t2
        
        A3 = A.copy(); b3 = b.copy()             
        t3 = time.time()
        res3 = ls_svdb(A3,b3,cond=1e-6,overwrite_a=False, overwrite_b=False)
        t3 = time.time() - t3
        
        d3 = np.max( res2[0] - res3[0] )
        
        times_2.append(t2)
        times_3.append(t3)
        diff3.append(d3)
        
    import matplotlib.pyplot as plt        
    
    plt.title('Algorihms speed comparison 2', size = 18, weight = 'bold')           
    plt.plot(m_steps, times_2, 'bo', label = 'COF')              
    plt.plot(m_steps, times_3, 'go', label = 'Block SVD')      
        
    plt.ylabel('Seconds', size = 15, weight = 'heavy' )
    plt.xlabel('Matrix m-size, n-size = 2000, rank = 1000 (if m>n )',\
                size = 15, weight = 'bold' )
    
    plt.legend(loc=2)
    plt.show()
    
    # Saved to ls_test_speed_comparison1.png 
    # and ls_test_accuracy_comparison1.png    
     
    return m_steps, times_2, times_3, diff3 


def test3():
    """
    Test how ls scales with respect to dimesion n, under fixed dimension
    m and rank k.
    
    Test only fastest algorithms COF and bSVD
    """
          
    import scipy as sp
    import math
    from numpy import random as rnd    
    import time
    
    times_2 = []    
    times_3 = []
    diff3 = []

    m = 10000 # Fixed second dimension          
    k = 1000 # Fixed rank  
    
    n_steps = range(100,1000,100 ) + range(1000, 16000, 1000)
    #n_steps = range(100,300,100)
        
    for (i,n) in enumerate( n_steps ):
        
        A =  10 * rnd.rand(m,k) - 5
        Temp =  10 * rnd.rand(k,n) - 5       
        
        A = A.dot(Temp)
        
        b =  10 * rnd.rand(m,1) - 5        
        
        A = np.asfortranarray(A)
        
                           
        A2 = A.copy(); b2 = b.copy()                  
        t2 = time.time()
        res2 = ls_cof(A2,b2,cond=1e-6,overwrite_a=False, overwrite_b=False)
        t2 = time.time() - t2
        
        A3 = A.copy(); b3 = b.copy()             
        t3 = time.time()
        res3 = ls_svdb(A3,b3,cond=1e-6,overwrite_a=False, overwrite_b=False)
        t3 = time.time() - t3
        
        d3 = np.max( res2[0] - res3[0] )
        
        times_2.append(t2)
        times_3.append(t3)
        diff3.append(d3)
        
    import matplotlib.pyplot as plt        
    
    plt.title('Algorihms speed comparison 3', size = 18, weight = 'bold')           
    plt.plot(n_steps, times_2, 'bo', label = 'COF')              
    plt.plot(n_steps, times_3, 'go', label = 'Block SVD')      
        
    plt.ylabel('Seconds', size = 15, weight = 'heavy' )
    plt.xlabel('Matrix n-size, m-size = 10000, rank = 1000 (if n>k )',\
                size = 15, weight = 'bold' )
    
    plt.legend(loc=2)
    plt.show()
    
    # Saved to ls_test_speed_comparison1.png 
    # and ls_test_accuracy_comparison1.png    
     
    return m_steps, times_2, times_3, diff3 


     
if __name__ == '__main__':
    ret = test3()   # For debuging in spider  
     
#    #m_steps = range(100,1000,100 ) + range(1000, 11000, 1000)
#    
#    # for (i,m) in enumerate( m_steps ):
#    m = 10000
#    #n  = math.ceil(2./3. * m)
#    n = 4500
#            
#    A =  10 * rnd.rand(m,n) - 5
#    C = 10 * rnd.rand(n, 5000)
#    A = A.dot(C)
#                   
#    b = 10 * rnd.rand(m,1) - 5
#    
#    
#    A = np.asfortranarray(A)
#    b = np.asfortranarray(b)
#    
#
#    A2 = A.copy(); b2 = b.copy()                  
#    t2 = time.time()
#    res2 = ls_cof(A2,b2,cond=1e-6,overwrite_a=False, overwrite_b=False)
#    t2 = time.time() - t2
#    
#    A3 = A.copy(); b3 = b.copy()             
#    t3 = time.time()
#    res3 = ls_svdb(A3,b3,cond=1e-6,overwrite_a=True, overwrite_b=True)
#    t3 = time.time() - t3
#        
##        A1 = A.copy(); b1 = b.copy()             
##        t1 = time.time()
##        res1 = ls_test.ls_svd(A1,b1,cond=1e-6,overwrite_a=True, overwrite_b=True)
##        t1 = time.time() - t1    
#        
##        times.append([t1,t2,t3])
#        
#        
##    from matplotlib import pyplot as plt    
##    
##    times_arr = np.asarray(times)
##    
    
    
    
