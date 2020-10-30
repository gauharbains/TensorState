import numpy as np
from scipy.stats import poisson
from scipy.optimize import linprog

"""
    This script consists of the algorithm presented in the paper :

    Paul Valiant and Gregory Valiant. Estimating the Unseen: Improved Estimators for Entropy and other Properties.
    Advances in Neural Information Processing Systems 26 (NIPS 2013)
  
"""

def makeFingerprint(sample):

    """This function returns the fingerprint of 
    a sample. The fingerprint is defined as the 
    histogram of histogram

    Args:
        sample : 1d array - sample

    Returns:
        fingerprint: fingerprint of the input 
    """

    bin_edges = [i+0.5 for i in range(int(min(sample)-1), int(max(sample)+1))]
    counts, edges =  np.histogram(sample, bin_edges)
    bin_edges = [i+0.5 for i in range(-1, int(max(counts)+1))]
    counts, edges =  np.histogram(counts, bin_edges)
    fingerprint = counts[1:]
    return fingerprint

def unseen(f):

    """ Input: fingerprint f, where f(i) represents number of elements 
    that appear i times in a sample. Thus sum_i i*f(i) = sample size.
    makeFinger function transforms a sample into the associated 
    fingerprint.
 
    Output: approximation of 'histogram' of true distribution. specifically, 
    histx(i) represents the number of domain elements that occur with
    probability x(i). Thus sum_i x(i)*histx(i) = 1, as distributions have
    total probability mass 1.

    Args:
        f : ingerprint of the sample

    Returns:
        histx, x: as descibed above
    """

    f = np.reshape(f, (1, f.shape[0]))
    g = np.array([ [ i for i in range(1,f.shape[1]+1)]])
    k = np.dot(f, np.array([[i for i in range(1,f.shape[1]+1)]]).T).squeeze()

    # algorithm parameters
    gridFactor = 1.1

    # the allowable discrepancy between the returned solution and the "best" (overfit).
    # Decreasing alpha increases the chances of overfitting
    alpha = 0.5
    # chances of overfitting 
    xLPmin = 1/(k*max(k,10))
    # max iteration for the linear program solver
    maxLPIters = 1000

    x = [0]
    histx = [0]
    flp = [0 for i in range(max(f.shape))]
  
    for i in range(1,max(f.shape)+1):
        if f[0,i-1] > 0:
            wind = [max(1, i - np.ceil(np.sqrt(i))), min(i + np.ceil(np.sqrt(i)), max(f.shape))] 
            wind = [int(i) for i in wind]
            if np.sum(f[0,wind[0]-1:wind[1]])< 2 * np.sqrt(i):
                x.append(i/k)
                histx.append(f[0,i-1])
                flp[i-1] = 0
            else:
                flp[i-1] = f[0,i-1]

    
    index = [ i if flp[i] > 0 else 0 for i in range(len(flp)) ]
    fmax = max(index)

    # If no LP portion, return the empirical histogram
    if len(index) == 0:
        return x[1:], histx[1:]

    # set up first LP
    # amount of probability mass in the LP region
    LPmass = 1 - sum([x*y for x,y in zip(x,histx)])
    flp = flp[:fmax+1] + [0 for i in range(int(np.ceil(np.sqrt(fmax+1))))]
    szLPf = len(flp)

    xLPmax = (fmax+1)/k
    power_rng = int(np.ceil(np.log(xLPmax/xLPmin)/np.log(gridFactor)) + 1)
    xLP = [xLPmin*gridFactor ** power for power in range(power_rng)]
    szLPx = len(xLP)

    objf = np.zeros((szLPx+2*szLPf,1))
    # discrepancy in ithfingerprint expectation, weighted by 1/sqrt(f(i) + 1)
    np.put(objf, [i for i in range(szLPx,objf.shape[0], 2 )], [1/i for i in [np.sqrt(j+1) for j in flp]])
    np.put(objf, [i for i in range(szLPx+1,objf.shape[0], 2 )], [1/i for i in [np.sqrt(j+1) for j in flp]])

    A = np.zeros((2*szLPf, szLPx + 2*szLPf))
    b = np.zeros((2*szLPf, 1))

    for i in range(szLPf):
        A[2*i, [i for i in range(szLPx)]] = [poisson.pmf(i+1,k*j) for j in xLP]
        A[2*i+1, [i for i in range(szLPx)]] = [-1*poisson.pmf(i+1,k*j) for j in xLP]
        A[2*i, szLPx+2*i] = -1
        A[2*i+1, szLPx+2*i+1]=-1
        b[2*i,0] = flp[i]
        b[2*i+1] = -1*flp[i]

    Aeq = np.zeros((1,szLPx+2*szLPf))
    Aeq[0,:szLPx] = xLP
    beq = LPmass

    for i in range(szLPx):
        # rescaling for better conditioning
        A[:,i] = A[:,i]/xLP[i]
        Aeq[0,i] = Aeq[0,i]/xLP[i]
    
    # first linear program
    output = linprog(objf, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=(0,np.inf),  options = {'maxiter': maxLPIters})
    fval = output.fun
    exitflag = output.status

    if exitflag == 1:
        print('Max iterations reached -- try increasing maxLPiters')
    elif exitflag == 2 or exitflag == 3:
        print('LP1 solution was not found, still solving LP2 anyway...')

    objf2 = 0 * objf
    objf2[0:szLPx,0] = 1

    A2 = np.vstack((A, objf.T))
    b2 = np.vstack((b,fval+alpha))

    for i in range(szLPx):
        # rescaling for better conditioning
        objf2[i,0] = objf2[i,0]/xLP[i]
    
    # second linear program to minimize support size
    output = linprog(objf2, A_ub=A2, b_ub=b2, A_eq=Aeq, b_eq=beq, bounds=(0,np.inf), method='revised simplex', options = {'maxiter': 1000, 'tol':1e-4})
    sol2 = output.x
    sol2 = [ round(i,7) for i in sol2]
    sol2 = np.array([sol2])
    exitflag2 = output.status

    if exitflag2 != 0:
        print('LP2 solutiion was not found')

    # append LP solution to empirical portion of histogram
    sol2[0, :szLPx] = np.divide(sol2[0, :szLPx], np.array([xLP]))
    x = x + xLP
    histx = histx + list(sol2[0])
    ind = np.argsort(x)
    x = sorted(x)
    histx = [histx[i] for i in ind ]
    ind = [i for i in range(len(histx)) if histx[i]>0 ]
    x = [x[i] for i in ind]
    histx = [histx[i] for i in ind]

    return histx, x
