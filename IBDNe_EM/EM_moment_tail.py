from scipy.special import logsumexp
import numpy as np
from plotting import *
from misc import *
from scipy.optimize import minimize
from scipy.ndimage.interpolation import shift
#from numba import jit
import sys

C = 2


#@jit(parallel=True)
def updatePosterior(N, bin1, bin2, bin_midPoint1, bin_midPoint2):
    #return updated T1 and T2
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))
    #print(sum_log_prob_not_coalesce)
    G = len(N)
    #calculate log probability of coalescing earlier than maxGen for T1
    alpha1 = bin_midPoint1/50 #this is a vector
    beta1 = 1-1/(2*N[-1]) #this is just a scalar
    temp1 = 1-beta1*np.exp(-alpha1)
    last_col_1 = sum_log_prob_not_coalesce[-1] + np.log(1-beta1) - alpha1*(1 + G) - np.log(2500) + np.log(G**2/temp1 + (2*G-1)/temp1**2 + 2/temp1**3)

    #calculate log probaability of coalescing earlier than maxGen for T2
    alpha2 = bin_midPoint2/50
    beta2 = 1-1/(2*N[-1])
    temp2 = 1-beta2*np.exp(-alpha2)
    last_col_2 = sum_log_prob_not_coalesce[-1] + np.log(1-beta2) - alpha2*(1 + G) - np.log(50) + np.log(G/temp2 + 1/temp2**2)

    #calculate the rest of the column
    log_g_over_50 = np.log(np.arange(1, G+1)/50)
    log_2_times_N_g = np.log(2*N)
    len_times_g_over_50_1 = bin_midPoint1.reshape((len(bin_midPoint1),1))@(np.arange(1, G+1).reshape((1, G)))/50
    len_times_g_over_50_2 = bin_midPoint2.reshape((len(bin_midPoint2),1))@(np.arange(1, G+1).reshape((1, G)))/50

    T1 = 2*log_g_over_50 - len_times_g_over_50_1 - log_2_times_N_g + sum_log_prob_not_coalesce[:-1]
    T2 = log_g_over_50 - len_times_g_over_50_2 - log_2_times_N_g + sum_log_prob_not_coalesce[:-1]

    #append one additional column of log likelihood of coalescing beyond maxGen generations into the past
    T1 = np.append(T1, last_col_1[:,np.newaxis], axis=1)
    T2 = np.append(T2, last_col_2[:, np.newaxis], axis=1)
    normalizing_constant1 = np.apply_along_axis(logsumexp, 1, T1)[:,np.newaxis]
    normalizing_constant2 = np.apply_along_axis(logsumexp, 1, T2)[:,np.newaxis]
    T1 = T1 - normalizing_constant1
    T2 = T2 - normalizing_constant2
    return T1, T2


#@jit(parallel=True)
def updateN(maxGen, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, n_p, log_term3, N, alpha, chr_len_cM):
    log_total_len_each_bin1 = np.log(bin1) + np.log(bin_midPoint1)
    log_total_len_each_bin2 = np.log(bin2) + np.log(bin_midPoint2)
    log_expected_ibd_len_each_gen1 = np.apply_along_axis(logsumexp, 0, T1 + log_total_len_each_bin1[:,np.newaxis])
    log_expected_ibd_len_each_gen2 = np.apply_along_axis(logsumexp, 0, T2 + log_total_len_each_bin2[:,np.newaxis])
    log_total_expected_ibd_len_each_gen = np.logaddexp(log_expected_ibd_len_each_gen1, log_expected_ibd_len_each_gen2)

    gen = np.arange(1, maxGen+1)
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))[:-1]
    log_numerator = np.log(n_p) + sum_log_prob_not_coalesce + np.log(0.5) - C*gen/50 + log_term3

    #a penalized optimization approach
    #gradientChecker(N, log_total_expected_ibd_len_each_gen, log_term3, n_p, alpha)
    bnds = [(1000, 10000000) for n in N]
    result = minimize(loss_func, N, args=(log_total_expected_ibd_len_each_gen, log_term3, n_p, alpha, chr_len_cM), 
                      method='L-BFGS-B',  bounds=bnds)
    print(result, flush=True)
    return result.x

def log_expectedIBD_beyond_maxGen_given_Ne(N, chr_len_cM, maxGen, n_p):
    def partB(g, N_g, maxGen, C, chromLen):
        part3 = np.sum((C*g/50 + 1)*chromLen) - len(chromLen)*(C**2)*g/50
        part2 = -C*g/50
        part1 = (g-maxGen-1)*np.log(1-1/(2*N_g))
        return np.exp(part1 + part2 + np.log(part3))
    N_past = N[-1]
    integral, err = quad(partB, maxGen+1, np.inf, args=(N_past, maxGen, C, chromLen))
    return np.log(n_p) - np.log(2*N_past) + np.sum(np.log(1-1/2*N)) + np.log(integral)




def loss_func(N, log_obs, log_term3, n_p, alpha, chr_len_cM):
    #print(f'calculate loss for N={N}')
    G = len(N)
    gen = np.arange(1, G+1)
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))[:-1]
    log_expectation = np.log(n_p) + sum_log_prob_not_coalesce - np.log(2*N) - C*gen/50 + log_term3
    np.append(log_expectation, log_expectedIBD_beyond_maxGen_given_Ne(N, chr_len_cM, G, n_p)) #need to calculate expected amount of IBD coalescing beyond maxGen generations into the past
    penalty = alpha*np.sum(np.diff(N, n=2)**2)
    diff_obs_expectation = np.exp(log_obs) - np.exp(log_expectation)
    return np.sum(diff_obs_expectation**2/np.exp(log_obs)) + penalty


def jacobian(N, log_obs, log_term3, n_p, alpha):
    maxGen = len(N)
    jacMatrix = np.zeros((maxGen, maxGen))

    #calculate diagonal elements
    gen = np.arange(1, maxGen+1)
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))[:-1]
    log_common_terms = np.log(n_p) + sum_log_prob_not_coalesce - C*gen/50 + log_term3
    np.fill_diagonal(jacMatrix, -np.exp(log_common_terms + np.log(0.5) -2*np.log(N)))

    #calculate lower triangular terms
    for g in range(2, maxGen+1):
        jacMatrix[g-1,:g-1] = np.exp(log_common_terms[g-1] - np.log(2*N[g-1]) - np.log(1-1/(2*N[:g-1])) 
                                     + np.log(0.5) - 2*np.log(N[:g-1]))

    #summing up
    log_expectation = log_common_terms - np.log(2*N)
    chain_part1 = 2*(np.exp(log_expectation)-np.exp(log_obs))/np.exp(log_obs)
    chi2_term = np.sum(jacMatrix*chain_part1[:,np.newaxis], axis=0)

    N_left2 = shift(N, -2, cval=0)
    N_left1 = shift(N, -1, cval=0)
    N_right2 = shift(N, 2, cval=0)
    N_right1 = shift(N, 1, cval=0)
    penalty_term = 12*N - 8*(N_left1 + N_right1) + 2*(N_left2 + N_right2)
    penalty_term[0] = 2*N[0]-4*N[1]+2*N[2]
    penalty_term[1] = 10*N[1]-4*N[0]-8*N[2]+2*N[3]
    penalty_term[-1] = 2*N[-1]-4*N[-2]+2*N[-3]
    penalty_term[-2] = 10*N[-2]-4*N[-1]-8*N[-3]+2*N[-4]

    return chi2_term + alpha*penalty_term


def testExpectation(maxGen, bin1, bin2, bin_midPoint1, bin_midPoint2):
    N1 = initializeN_Uniform(maxGen, 10000)
    N2 = initializeN_Uniform(maxGen, 1000)
    N3 = initializeN_Uniform(maxGen, 100)
    T1_1, T2_1 = updatePosterior(N1, bin1, bin2, bin_midPoint1, bin_midPoint2)
    T1_2, T2_2 = updatePosterior(N2, bin1, bin2, bin_midPoint1, bin_midPoint2)
    T1_3, T2_3 = updatePosterior(N3, bin1, bin2, bin_midPoint1, bin_midPoint2)
    print(T1_1.shape)
    print(np.exp(T1_1.T))
    print(np.exp(T1_2.T))
    print(np.exp(T1_3.T))


#test gradient calculation for loss_func in EMbyMoment.py
def gradientChecker(N, log_obs, log_term3, n_p, alpha):
    delta = 1e-6
    maxGen = N.size
    calculated = jacobian(N, log_obs, log_term3, n_p, alpha)
    gradient = np.zeros(maxGen)
    for g in np.arange(maxGen):
        #print(np.eye(maxGen)[g])
        upper, lower = loss_func(N + delta*np.eye(maxGen)[g], log_obs, log_term3, n_p, alpha), loss_func(N - delta*np.eye(maxGen)[g], log_obs, log_term3, n_p, alpha)
        gradient[g] = (upper - lower)/(2*delta)
        print(f'diff between up and low: {upper-lower}')
    print(f'calculated gradient is: {calculated}')
    print(f'approximated gradient is: {gradient}')
    sys.exit()


def em_moment_tail(maxGen, bin1, bin2, bin_midPoint1, bin_midPoint2, chr_len_cM, numInds, alpha, tol, maxIter):
    N = initializeN_autoreg(maxGen)
    #N = initializeN_Uniform(maxGen, 20000)
    print(f"initial N:{N}", flush=True)

    #pre-calculate log of term3 in the updateN step
    #this quantity is a constant in all iterations
    #so is more efficient to calculate here, save as a local variable, and pass it onto future iterations
    n_p = (2*numInds)*(2*numInds-2)/2
    chr_len_cM = chr_len_cM[:,np.newaxis]
    gen = np.arange(1, maxGen+1).reshape((1, maxGen))
    log_term3 = np.log(np.sum(C*(chr_len_cM@gen)/50 + chr_len_cM - ((C**2)*gen)/50, axis=0))

    #data preprocessing done. Start EM.
    N_prev = N
    T1, T2 = updatePosterior(N, bin1, bin2, bin_midPoint1, bin_midPoint2)
    N = updateN(maxGen, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, n_p, log_term3, N, alpha, chr_len_cM)
    N_curr = N
    num_iter = 1
    diff = N_curr - N_prev
    dist = diff.dot(diff)/maxGen

    while ( dist >= tol and num_iter < maxIter):
        print(f'iteration{num_iter} done. Diff: {dist}', flush=True)
        N_prev = N_curr
        T1, T2 = updatePosterior(N, bin1, bin2, bin_midPoint1, bin_midPoint2)
        N = updateN(maxGen, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, n_p, log_term3, N, alpha, chr_len_cM)
        N_curr = N
        diff = N_curr - N_prev
        dist = diff.dot(diff)/maxGen
        num_iter += 1
    
    print(f'iteration{num_iter} done. Diff: {dist}', flush=True)
    plotPosterior(np.exp(T1.T), bin_midPoint1, np.arange(1, maxGen+1), title=f'Posterior Distribution for Iteration {num_iter}')    
    return N, T1, T2

