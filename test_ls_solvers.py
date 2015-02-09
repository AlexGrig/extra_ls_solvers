# -*- coding: utf-8 -*-
"""
author: Alexander Grigorevskiy, 2013

This module contain some performance tests of alternative least-squares solvers
provided in extra_ls_solvers module.
"""

import numpy as np
import extra_ls_solvers

def test1():
    """
    Test the speed of four ls solvers on not full rank matrices.
    Also keep track of difference in the solutions.

    The matrix has the size (m,2/3*m) the rank is 1/2*m.
    Matrix values are random in the range (-5, 5), the same is for right hand
    side.
    """

    import matplotlib.pyplot as plt
    import scipy.io as io
    import math
    from numpy import random as rnd
    import time

    times_1 = [] # Current Scipy (SVD) (gelss)
    times_2 = [] # COF (gelsy)
    times_3 = [] # Block SVD (gelsd)
    times_4 = [] # Standard Numpy
    diff1 = []
    diff2 = []
    diff3 = []
    diff4 = []

    m_steps = range(100,1000,100 ) + range(1000, 11000, 1000)
    #m_steps = range(100,300,100)
    for (i,m) in enumerate( m_steps ):
        print m

        n  = math.ceil(2./3. * m )
        k = math.ceil(1./2. * m )

        A =  10 * rnd.rand(m,k) - 5
        Temp =  10 * rnd.rand(k,n) - 5

        A = A.dot(Temp)

        b =  10 * rnd.rand(m,1) - 5

        A = np.asfortranarray(A) # with fortran object the functions should run
                                 # faster

        A1 = A.copy(); b1 = b.copy()
        t1 = time.time()
        res1 = extra_ls_solvers.ls_scipy_current_solver(A1,b1,cond=1e-6,overwrite_a=False, overwrite_b=False)
        t1 = time.time() - t1

        A2 = A.copy(); b2 = b.copy()
        t2 = time.time()
        res2 = extra_ls_solvers.ls_cof(A2,b2,cond=1e-6,overwrite_a=False, overwrite_b=False)
        t2 = time.time() - t2

        d1 = np.max( res1[0] - res2[0] )

        A3 = A.copy(); b3 = b.copy()
        t3 = time.time()
        res3 = extra_ls_solvers.ls_svdb(A3,b3,cond=1e-6,overwrite_a=False, overwrite_b=False)
        t3 = time.time() - t3

        d2 = np.max( res1[0] - res3[0] )
        d3 = np.max( res2[0] - res3[0] )

        A4 = A.copy(); b4 = b.copy()
        t4 = time.time()
        res4 = extra_ls_solvers.ls_numpy_standard_solver(A4,b4,cond=1e-6)
        t4 = time.time() - t4

        d4 = np.max( res2[0] - res4[0] )

        times_1.append(t1)
        times_2.append(t2)
        times_3.append(t3)
        times_4.append(t4)

        diff1.append(d1)
        diff2.append(d2)
        diff3.append(d3)
        diff4.append(d4)

    res_dict = {'m_steps': m_steps, 'times_1': times_1, 'times_2':  times_2,
                'times_3': times_3, 'times_4': times_4, 'diff1': diff1,
                'diff2': diff2, 'diff3': diff3, 'diff4': diff4 }

    io.savemat('ls_test_1_results.mat',  res_dict )

    # Saving results to ls_test_1_speeds.png
    # and ls_test_1_accuracies.png
    plt.figure(1)
    plt.title('Algorihms Speed Comparison', size = 18, weight = 'bold')

    plt.plot(m_steps, times_1, 'ro', label = 'Current Scipy (SVD) (gelss)')
    plt.plot(m_steps, times_2, 'bo', label = 'COF (gelsy)')
    plt.plot(m_steps, times_3, 'go', label = 'Block SVD (gelsd)')
    plt.plot(m_steps, times_4, 'yo', label = 'Standard Numpy')

    plt.ylabel('Seconds', size = 15, weight = 'heavy' )
    plt.xlabel('Matrix m-size, n-size = 2/3 m-size, rank = 1/2 m-size',\
                size = 15, weight = 'bold' )

    plt.legend(loc=2)
    plt.savefig('ls_test_1_speeds.png')

    plt.close()
    plt.figure(2)

    plt.title('Algorihms Accuracy Comparison', size = 18, weight = 'bold')

    plt.plot(m_steps, diff1, 'ro', label = 'Current Scipy (SVD) (gelss)')
    plt.plot(m_steps, diff2, 'bo', label = 'COF (gelsy)')
    plt.plot(m_steps, diff3, 'go', label = 'Block SVD (gelsd)')
    plt.plot(m_steps, diff4, 'yo', label = 'Standard Numpy')

    plt.ylabel('Seconds', size = 15, weight = 'heavy' )
    plt.xlabel('Matrix m-size, n-size = 2/3 m-size, rank = 1/2 m-size',\
                size = 15, weight = 'bold' )

    plt.legend(loc=2)
    plt.savefig('ls_test_1_accuracies.png')

    return m_steps, times_1, times_2, times_3, times_4, diff1, diff2, diff3, \
            diff4

def test2(option=2):
    """
    Test how ls solvers scale with respect to one dimesion, under the fixed
    other dimension and rank k.

    The matrix has size (m,n) the rank is k.
    Matrix values are random in the range (-5, 5), the same is for right hand
    side.

    Inputs:
        option - which type of test is performed
                 2: m changes, n=2000 and k=1000 are fixed
                 3: n changes, m=10000 and k=1000 are fixed
    """

    import matplotlib.pyplot as plt
    import scipy.io as io
    from numpy import random as rnd
    import time

    times_1 = [] # Current Scipy (SVD) (gelss)
    times_2 = [] # COF (gelsy)
    times_3 = [] # Block SVD (gelsd)
    diff1 = []
    diff2 = []

    if (option == 2):
        #steps = range(100,1000,100 ) + range(1000, 12000, 1000)
        steps = range(1000, 120000, 1000)
    elif (option == 3):
        #steps = range(100,1000,100 ) + range(1000, 12000, 1000)
        steps = range(1000, 120000, 1000)
    else:
        raise ValueError( "option = %s is not supported" % (option, ) )

    #steps = range(100,300,100)

    for (i,s) in enumerate( steps ):
        print s

        if (option == 2):
            m = s
            n = 500 #2000# Fixed second dimension
            k = 480 #1000# Fixed rank
        elif (option == 3):
            n = s
            m = 500 #10000# Fixed second dimension
            k = 480 #1000# Fixed rank

        A =  10 * rnd.rand(m,k) - 5
        Temp =  10 * rnd.rand(k,n) - 5

        A = A.dot(Temp)

        b =  10 * rnd.rand(m,1) - 5


        A = np.asfortranarray(A) # with fortran object the functions should run
                                 # faster

        A1 = A.copy(); b1 = b.copy()
        t1 = time.time()
        res1 = extra_ls_solvers.ls_scipy_current_solver(A1,b1,cond=1e-6,overwrite_a=False, overwrite_b=False)
        t1 = time.time() - t1

        A2 = A.copy(); b2 = b.copy()
        t2 = time.time()
        res2 = extra_ls_solvers.ls_cof(A2,b2,cond=1e-6,overwrite_a=False, overwrite_b=False)
        t2 = time.time() - t2

        A3 = A.copy(); b3 = b.copy()
        t3 = time.time()
        res3 = extra_ls_solvers.ls_svdb(A3,b3,cond=1e-6,overwrite_a=False, overwrite_b=False)
        t3 = time.time() - t3

        d1 = np.max( res1[0] - res2[0] )
        d2 = np.max( res1[0] - res3[0] )

        times_1.append(t1)
        times_2.append(t2)
        times_3.append(t3)
        diff1.append(d1)
        diff2.append(d2)

    res_dict = {'option': option,'steps': steps, 'times_1': times_1, 'times_2':  times_2,
                'times_3': times_3, 'diff1': diff1,
                'diff2': diff2 }
    save_file_name = 'ls_test_%i_results.mat' % (option+3)
    io.savemat(save_file_name,  res_dict )


    plt.figure(1)
    if (option == 2):
        graph_title = 'Algorihms Speed Comparison 2 (fixed n,k)'
        x_axis_title = 'Matrix m-size, n-size = %i, rank = %i (if m>n )' % (n,k )
    elif (option == 3):
        graph_title = 'Algorihms Speed Comparison 3 (fixed m,k)'
        x_axis_title = 'Matrix n-size, m-size = %i, rank = %i (if n>k )' % (m,k)

    plt.title(graph_title, size = 18, weight = 'bold')
    plt.plot(steps, times_1, 'ro', label = 'Current Scipy (SVD) (gelss)')
    plt.plot(steps, times_2, 'bo', label = 'COF (gelsy)')
    plt.plot(steps, times_3, 'go', label = 'Block SVD (gelsd)')

    plt.ylabel('Seconds', size = 15, weight = 'heavy' )
    plt.xlabel(x_axis_title, size = 15, weight = 'bold' )

    plt.legend(loc=2)
    save_file_name = 'ls_test_%i_speeds.png' % (option+3)
    #plt.show()
    plt.savefig(save_file_name)

    plt.close()
    plt.figure(2)
    if (option == 2):
        graph_title = 'Accuracy Comparison 2 (fixed n,k)'
    elif (option == 3):
        graph_title = 'Accuracy Comparison 3 (fixed m,k)'

    plt.title(graph_title, size = 18, weight = 'bold')

    plt.plot(steps, diff1, 'ro', label = 'Current Scipy - COF')
    plt.plot(steps, diff2, 'bo', label = 'Current Scipy - Block SVD')

    plt.ylabel('max-norm', size = 15, weight = 'heavy' )
    plt.xlabel(x_axis_title, size = 15, weight = 'bold' )

    plt.legend(loc=2)
    save_file_name = 'ls_test_%i_accuracies.png' % (option+3)
    #plt.show()
    plt.savefig(save_file_name)

    return steps, times_1, times_2, times_3, diff1, diff2


if __name__ == '__main__':
    #test1()
    #test2(3)
    test2(3)
