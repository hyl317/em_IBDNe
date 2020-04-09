from scipy.special import logsumexp
import numpy as np
from misc import initializeN_autoreg
from scipy.optimize import minimize
from scipy.ndimage.interpolation import shift
from plotting import *
import sys

def updatePosterior(IBD_count_by_bin, bins, maxGen, N):
    #calculate log P(t=g|Ne)
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))[:-1]
    log_P_g_given_N = np.zeros(maxGen)
    log_P_g_given_N = sum_log_prob_not_coalesce - np.log(2*N)

    #calculate log P(bin range | coalesce at g)
    #calculate P(u,\infty|g)
    g = np.arange(1, maxGen+1)
    tmp = g[:,np.newaxis]@bins.reshape((1, len(bins)))/50
    log_P_greater_than_u_given_g = -tmp + np.log(1+tmp)
    log_P_greater_than_u_given_g_shifted = np.apply_along_axis(shift, 1, log_P_greater_than_u_given_g, -1, cval=-np.inf)
    #calculate P(u,v|g)
    log_P_range_given_g = np.log(np.exp(log_P_greater_than_u_given_g)-np.exp(log_P_greater_than_u_given_g_shifted))

    #add the last row to T
    N_G = N[-1]
    alpha = np.log(1-1/(2*N_G)) - bins/50
    last_row = sum_log_prob_not_coalesce[-1] - np.log(100*N_G) + (maxGen+1)*np.log((2*N_G)/(2*N_G-1)) + alpha*(maxGen+1) + np.log(bins + (1-np.exp(alpha))*(50 + bins*maxGen)) - 2*np.log(1-np.exp(alpha))
    last_row_shifted = shift(last_row, -1, cval=-np.inf)
    last_row = np.log(np.exp(last_row)-np.exp(last_row_shifted))

    T = np.zeros((maxGen+1, len(bins)))
    T[:maxGen, :] = log_P_range_given_g + log_P_g_given_N[:,np.newaxis]
    T[maxGen] = last_row
    #normalize T
    normalizing_constant = np.apply_along_axis(logsumexp, 0, T)
    return T - normalizing_constant


def em_by_count(IBD_count_by_bin, bins, total_genome_length, numInds, maxGen, alpha, maxIter, tol):
    N = initializeN_autoreg(maxGen, 1e5)
    print(f"initial N:{N}", flush=True)

    numIter = 1
    T = updatePosterior(IBD_count_by_bin, bins, maxGen, N)
    tmp = np.apply_along_axis(logsumexp, 0, T)
    print(np.exp(tmp))
    plotPosterior(np.exp(T), bins, np.arange(1, maxGen+1), title=f'Posterior Distribution for Iteration {numIter}')    
    sys.exit()
    N = updateN(T, IBD_count_by_bin, bins, maxGen, alpha)

    while numIter < maxIter:
        T = updatePosterior(IBD_count_by_bin, bins, maxGen, N)
        N = updateN(T, IBD_count_by_bin, bins, maxGen, alpha)
        numIter += 1

    return N, T
