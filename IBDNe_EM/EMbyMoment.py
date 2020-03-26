from scipy.special import logsumexp
import numpy as np
from plotting import *
from misc import *
from scipy.optimize import minimize
from scipy.ndimage.interpolation import shift
import sys

C = 2

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

def updateN(maxGen, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, n_p, log_term3, N, alpha):
    log_total_len_each_bin1 = np.log(bin1) + np.log(bin_midPoint1)
    log_total_len_each_bin2 = np.log(bin2) + np.log(bin_midPoint2)
    log_expected_ibd_len_each_gen1 = np.apply_along_axis(logsumexp, 0, T1 + log_total_len_each_bin1[:,np.newaxis])
    log_expected_ibd_len_each_gen2 = np.apply_along_axis(logsumexp, 0, T2 + log_total_len_each_bin2[:,np.newaxis])
    log_total_expected_ibd_len_each_gen = np.logaddexp(log_expected_ibd_len_each_gen1, log_expected_ibd_len_each_gen2)

    gen = np.arange(1, maxGen+1)
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))[:-1]
    log_numerator = np.log(n_p) + sum_log_prob_not_coalesce + np.log(0.5) - C*gen/50 + log_term3

    #point_fit_N = np.exp(log_numerator - log_total_expected_ibd_len_each_gen)
    #the following two lines implement the method as it is in the original IBDNe paper    
    #exp_fit_N = fit_exp_curve(log_numerator, log_total_expected_ibd_len_each_gen)
    #powell_fit_N = fmin_powell(loss_func, N, args=(log_total_expected_ibd_len_each_gen, log_term3, n_p, alpha), disp=1, retall=0, xtol=1e-4, ftol=1e-2)
    #print(f'loss after point fitting is {loss_func(point_fit_N, log_total_expected_ibd_len_each_gen, log_term3, n_p, alpha)}')
    #print(f'loss after piecewise exp fitting is {loss_func(exp_fit_N, log_total_expected_ibd_len_each_gen, log_term3, n_p, alpha)}')
    #print(f'loss after powell fitting is {loss_func(powell_fit_N, log_total_expected_ibd_len_each_gen, log_term3, n_p, alpha)}') 
    #print(f'powell fitting: {powell_fit_N}')
    #return powell_fit_N

    #a penalized optimization approach
    #gradientChecker(N, log_total_expected_ibd_len_each_gen, log_term3, n_p, alpha)
    bnds = [(1000, 10000000) for n in N]
    result = minimize(loss_func, N, args=(log_total_expected_ibd_len_each_gen, log_term3, n_p, alpha), 
                      method='L-BFGS-B',  bounds=bnds, jac=jacobian)
    print(result, flush=True)
    return result.x


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
    #print(f'calculate loss for N={N}')
    G = len(N)
    gen = np.arange(1, G+1)
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))[:-1]
    log_expectation = np.log(n_p) + sum_log_prob_not_coalesce - np.log(2*N) - C*gen/50 + log_term3
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
    #print(f'chain part 1: {chain_part1}')
    #print(f'log_term3 is {log_term3}')
    #print(f'obs is {np.exp(log_obs)}')
    #print(f'exp is {np.exp(log_expectation)}')
    #print(f'jac matrix is: {jacMatrix}')
    #penalty for roughness(second difference)
    N_left2 = shift(N, -2, cval=0)
    N_left1 = shift(N, -1, cval=0)
    N_right2 = shift(N, 2, cval=0)
    N_right1 = shift(N, 1, cval=0)
    penalty_term = 12*N - 8*(N_left1 + N_right1) + 2*(N_left2 + N_right2)
    penalty_term[0] = 2*N[0]-4*N[1]+2*N[2]
    penalty_term[1] = 10*N[1]-4*N[0]-8*N[2]+2*N[3]
    penalty_term[-1] = 2*N[-1]-4*N[-2]+2*N[-3]
    penalty_term[-2] = 10*N[-2]-4*N[-1]-8*N[-3]+2*N[-4]
    #print(f'penalty component is {alpha*penalty_term}')
    return chi2_term + alpha*penalty_term


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



def em_byMoment(maxGen, bin1, bin2, bin_midPoint1, bin_midPoint2, chr_len_cM, numInds, alpha, tol, maxIter):
    N = initializeN_autoreg(maxGen)
    #N = initializeN_Uniform(maxGen, 20000)
    print(f"initial N:{N}")

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
    N = updateN(maxGen, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, n_p, log_term3, N, alpha)
    #T1, T2 = updatePosterior(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2)
    N_curr = N
    num_iter = 1
    diff = N_curr - N_prev
    dist = diff.dot(diff)/maxGen
    #plotPosterior(np.exp(T1.T), bin_midPoint1, np.arange(1, maxGen+1), title=f'Posterior Distribution for Iteration {num_iter}')

    while ( dist >= tol and num_iter < maxIter):
        print(f'iteration{num_iter} done. Diff:{dist}')
        N_prev = N_curr
        T1, T2 = updatePosterior(N, bin1, bin2, bin_midPoint1, bin_midPoint2)
        N = updateN(maxGen, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, n_p, log_term3, N, alpha)
        N_curr = N
        diff = N_curr - N_prev
        dist = diff.dot(diff)/maxGen
        num_iter += 1
    
    print(f'iteration{num_iter} done.')
    plotPosterior(np.exp(T1.T), bin_midPoint1, np.arange(1, maxGen+1), title=f'Posterior Distribution for Iteration {num_iter}')    
    return N, T1, T2
