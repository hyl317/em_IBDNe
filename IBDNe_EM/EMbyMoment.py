from scipy.special import logsumexp
import numpy as np
from plotting import *
from misc import *
from csaps import csaps
from scipy.optimize import minimize

C = 2


#def initializeT_Random(numBins, maxGen):
#    T = np.random.rand(numBins, maxGen)
#    return T/T.sum(axis=1)[:, np.newaxis]


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

    #this is still log of unnormalized probabilities
    #is normalization necessary?
    normalizing_constant1 = np.logaddexp(np.apply_along_axis(logsumexp, 1, T1)[:,np.newaxis], last_col_1[:, np.newaxis])
    normalizing_constant2 = np.logaddexp(np.apply_along_axis(logsumexp, 1, T2)[:,np.newaxis], last_col_2[:, np.newaxis])
    T1 = T1 - normalizing_constant1
    T2 = T2 - normalizing_constant2
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

    #the following two lines implement the method as it is in the original IBDNe paper    
    final_N = fit_exp_curve(log_numerator, log_total_expected_ibd_len_each_gen)
    return final_N

    #a penalized optimization approach
    #bnds = [(0, np.inf) for n in N]
    #result = minimize(loss_func, N, args=(log_total_expected_ibd_len_each_gen, log_term3, n_p, 0.05), 
    #                  method='L-BFGS-B', tol=1e-6, bounds=bnds)
    #print(result)
    #return result.x


    #a spline approach (not quite right)
    #log_N_updated = log_numerator - log_total_expected_ibd_len_each_gen
    #final_N = csaps(np.arange(0, maxGen), np.exp(log_N_updated), np.arange(0, maxGen), smooth=0.8)    
    #return np.exp(log_N_updated)
    #return final_N

def fn(r, X, Y, prev, interval):
    exponent = np.arange(-interval,0,1)
    return np.sum(X)-np.sum(Y*np.exp(r*exponent))/prev

def Dfn(r, X, Y, prev, interval):
    exponent = np.arange(-interval,0,1)
    return -np.sum(exponent*np.exp(r*exponent)*Y)/prev

def loss_func(N, log_obs, log_term3, n_p, alpha):
    G = len(N)
    gen = np.arange(1, G+1)
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))[:-1]
    log_expectation = np.log(n_p) + sum_log_prob_not_coalesce + np.log(0.5) - C*gen/50 + log_term3
    
    N_shifted = np.roll(N,-1)
    N_shifted[-1] = N[-1]
    diff = N_shifted - N
    penalty = alpha*np.sum(np.dot(diff, diff))

    diff_obs_expectation = np.exp(log_obs) - np.exp(log_expectation)
    #print(f'diff between obs and expected:{diff_obs_expectation}')
    return np.sum(np.dot(diff_obs_expectation, diff_obs_expectation))/G + penalty

def jacobian(N, log_obs, log_term3, n_p, alpha):
    G = len(N)

    gen = np.arange(1, G+1)
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))[:-1]
    log_expectation = np.log(n_p) + sum_log_prob_not_coalesce + np.log(0.5) - C*gen/50 + log_term3
    residual_term = 2*(np.exp(log_obs)-np.exp(log_expectation))*np.exp(log_expectation)/(G*N)

    #penalty for roughness
    N_left = np.roll(N,-1)
    N_right = np.roll(N,1)
    penalty_term = 4*N - 2*(N_left + N_right)
    penalty_term[0] = 2*(N[0] - N[1])
    penalty_term[-1] = 2*(N[-1] - N[-2])

    return residual_term + alpha*penalty_term


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
        r = newton(fn, Dfn, 0, 1e-4, 200, Xs, Ys, prev, interval)
        if r == None or abs(r) >= 2:
            final_N[maxGen-i*interval:maxGen-(i-1)*interval] = Ys/Xs
        else:
            final_N[maxGen-i*interval:maxGen-(i-1)*interval] = prev*np.exp(r*np.arange(interval,0,-1))
    
    final_N = csaps(np.arange(0, maxGen), final_N, np.arange(0, maxGen), smooth=0.8)
    return final_N

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

def em_byMoment(maxGen, bin1, bin2, bin_midPoint1, bin_midPoint2, chr_len_cM, numInds, tol, maxIter):
    #N = initializeN_autoreg(maxGen)
    #testExpectation(maxGen, bin1, bin2, bin_midPoint1, bin_midPoint2)
    N = initializeN_autoreg(maxGen)
    print(f"initial N:{N}")

    #pre-calculate log of term3 in the updateN step
    #this quantity is a constant in all iterations
    #so is more efficient to calculate here, save as a local variable, and pass it onto future iterations
    n_p = (2*numInds)*(2*numInds-2)/2
    #print(chr_len_cM)
    chr_len_cM = chr_len_cM[:,np.newaxis]
    gen = np.arange(1, maxGen+1).reshape((1, maxGen))
    log_term3 = np.log(np.sum(C*(chr_len_cM@gen)/50 + chr_len_cM - ((C**2)*gen)/50, axis=0))

    #data preprocessing done. Start EM.
    N_prev = N
    T1, T2 = updatePosterior(N, bin1, bin2, bin_midPoint1, bin_midPoint2)
    N = updateN(maxGen, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, n_p, log_term3, N)
    #T1, T2 = updatePosterior(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2)
    N_curr = N
    num_iter = 1
    diff = N_curr - N_prev
    dist = diff.dot(diff)
    plotPosterior(np.exp(T1.T), bin_midPoint1, np.arange(1, maxGen+1), title=f'Posterior Distribution for Iteration {num_iter}')

    while ( dist >= tol and num_iter < maxIter):
        print(f'iteration{num_iter} done. Diff:{dist}')
        N_prev = N_curr
        T1, T2 = updatePosterior(N, bin1, bin2, bin_midPoint1, bin_midPoint2)
        N = updateN(maxGen, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, n_p, log_term3, N)
        N_curr = N
        diff = N_curr - N_prev
        dist = diff.dot(diff)
        num_iter += 1
    
    print(f'iteration{num_iter} done.')
    plotPosterior(np.exp(T1.T), bin_midPoint1, np.arange(1, maxGen+1), title=f'Posterior Distribution for Iteration {num_iter}')    
    return N, T1, T2
