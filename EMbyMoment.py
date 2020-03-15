from EM import initializeN
from scipy.special import logsumexp
import numpy as np
from plotting import *
from misc import *
#from scipy.interpolate import UnivariateSpline
#from csaps import csaps


NUM_INDS = 1000
C = 2

def refFinNe():
    growth_rate1 = 0.0247
    growth_rate2 = 0.182
    N_0 = 1000
    N_curr = N_0
    N = [N_0]
    for g in np.arange(99, 0, -1):
        if g >= 13:
            N_curr = N_curr*(np.exp(growth_rate1))
        else:
            N_curr = N_curr*(np.exp(growth_rate2))
        N.insert(0, N_curr)
    return np.array(N)


def initializeT_Random(numBins, maxGen):
    T = np.random.rand(numBins, maxGen)
    return T/T.sum(axis=1)[:, np.newaxis]

def updatePosterior(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2):
    #return updated T1 and T2
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))

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
    log_g_over_50 = np.arange(1, G+1) - np.log(50)
    log_2_times_N_g = np.log(2*N)
    len_times_g_over_50_1 = bin_midPoint1.reshape((len(bin_midPoint1),1))@(np.arange(1, G+1).reshape((1, G)))/50
    len_times_g_over_50_2 = bin_midPoint2.reshape((len(bin_midPoint2),1))@(np.arange(1, G+1).reshape((1, G)))/50

    T1 = 2*log_g_over_50 - len_times_g_over_50_1 - log_2_times_N_g + sum_log_prob_not_coalesce[:-1]
    T2 = log_g_over_50 - len_times_g_over_50_2 - log_2_times_N_g + sum_log_prob_not_coalesce[:-1]

    #this is still log of unnormalized probabilities
    #is normalization necessary?
    normalizing_constant1 = np.logaddexp(np.apply_along_axis(logsumexp, 1, T1)[:,np.newaxis], last_col_1[:, np.newaxis])
    normalizing_constant2 = np.logaddexp(np.apply_along_axis(logsumexp, 1, T2)[:,np.newaxis], last_col_2[:, np.newaxis])
    T1 = T1 - normalizing_constant1
    T2 = T2 - normalizing_constant2
    #print(np.sum(np.exp(T1),axis=1))
    #print(np.sum(np.exp(T2),axis=1))
    return T1, T2

def updateN(maxGen, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, n_p, log_term3, N):
    log_total_len_each_bin1 = np.log(bin1) + np.log(bin_midPoint1)
    log_total_len_each_bin2 = np.log(bin2) + np.log(bin_midPoint2)
    log_expected_ibd_len_each_gen1 = np.apply_along_axis(logsumexp, 0, T1 + log_total_len_each_bin1[:,np.newaxis])
    log_expected_ibd_len_each_gen2 = np.apply_along_axis(logsumexp, 0, T2 + log_total_len_each_bin2[:,np.newaxis])
    log_total_expected_ibd_len_each_gen = np.logaddexp(log_expected_ibd_len_each_gen1, log_expected_ibd_len_each_gen2)

    gen = np.arange(1, maxGen+1)
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))[:-1]
    log_numerator = np.log(n_p) + sum_log_prob_not_coalesce + np.log(0.5) - C*gen/50 + log_term3
    
    final_N = fit_exp_curve(log_numerator, log_total_expected_ibd_len_each_gen)
    print(final_N)
    return final_N
    #log_N_updated = log_numerator - log_total_expected_ibd_len_each_gen
    #return np.exp(log_N_updated)

def fn(r, X, Y, prev, interval):
    exponent = np.arange(-interval,0,1)
    return np.sum(X)-np.sum(Y*np.exp(r*exponent))/prev

def Dfn(r, X, Y, prev, interval):
    exponent = np.arange(-interval,0,1)
    return -np.sum(exponent*np.exp(r*exponent)*Y)/prev


def fit_exp_curve(log_numerator, log_denominator, interval=10):
    #for the last interval, assume a constant Ne
    assert len(log_numerator) == len(log_denominator)
    maxGen = len(log_numerator)
    final_N = np.zeros(maxGen)
    N = np.exp(logsumexp(log_numerator[maxGen-interval:]) - logsumexp(log_denominator[maxGen-interval:]))
    final_N[maxGen-interval:] = N

    TOTAL_NUM_INTERVALS = int(maxGen/interval)
    #print(TOTAL_NUM_INTERVALS)
    for i in range(2, TOTAL_NUM_INTERVALS+1):
        #calculate the interval [maxGen-i*interval, maxGen-(i-1)*interval)
        Ys = np.exp(log_numerator[maxGen-i*interval:maxGen-(i-1)*interval])
        Xs = np.exp(log_denominator[maxGen-i*interval:maxGen-(i-1)*interval])
        prev = final_N[maxGen-(i-1)*interval]
        r = newton(fn, Dfn, 0, 1e-4, 100, Xs, Ys, prev, interval)
        print(Ys)
        print(Xs)
        if r == None:
            final_N[maxGen-i*interval:maxGen-(i-1)*interval] = Ys/Xs
        else:
            final_N[maxGen-i*interval:maxGen-(i-1)*interval] = prev*np.exp(r*np.arange(1, interval+1,1))
    return final_N



def em_byMoment(maxGen, bin1, bin2, bin_midPoint1, bin_midPoint2, chr_len_cM, tol, maxIter):
    N, T1, T2 = initializeN(maxGen), initializeT_Random(bin1.shape[0], maxGen), initializeT_Random(bin2.shape[0], maxGen)
    print(f"initial N:{N}")

    #pre-calculate log of term3 in the updateN step
    #this quantity is a constant in all iterations
    #so is more efficient to calculate here, save as a local variable, and pass it onto future iterations
    n_p = (2*NUM_INDS)*(2*NUM_INDS-2)/2
    #print(chr_len_cM)
    chr_len_cM = chr_len_cM[:,np.newaxis]
    gen = np.arange(1, maxGen+1).reshape((1, maxGen))
    log_term3 = np.log(np.sum(C*(chr_len_cM@gen)/50 + chr_len_cM - ((C**2)*gen)/50, axis=0))

    #data preprocessing done. Start EM.
    N_prev = N
    T1, T2 = updatePosterior(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2)
    N = updateN(maxGen, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, n_p, log_term3, N)
    N_curr = N
    num_iter = 1
    diff = N_curr - N_prev
    dist = diff.dot(diff)
    plotPosterior(np.exp(T1.T), bin_midPoint1, np.arange(1, maxGen+1), title=f'Posterior Distribution for Iteration {num_iter}')

    while ( dist >= tol and num_iter < maxIter):
        print(f'iteration{num_iter} done. Diff:{dist}')
        N_prev = N_curr
        T1, T2 = updatePosterior(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2)
        N = updateN(maxGen, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, n_p, log_term3, N)
        N_curr = N
        diff = N_curr - N_prev
        dist = diff.dot(diff)
        num_iter += 1
    
    print(f'iteration{num_iter} done.')
    plotPosterior(np.exp(T1.T), bin_midPoint1, np.arange(1, maxGen+1), title=f'Posterior Distribution for Iteration {num_iter}')    
    return N, T1, T2
