#primary functions for EM algorithm
import numpy as np
import math
import statsmodels.api as sm
from scipy.special import logsumexp

def initializeN(maxGen):
    #initialize N, the population size trajectory
    phi = 0.98
    ar = np.array([1, -phi])
    MU, sigma = math.log(10000), math.log(10000)/10 
    #specify a AR(1) model, by default the mean 0 since it's a stationary time series
    N = sm.tsa.arma_generate_sample(ar, np.array([1]), maxGen+1, scale=math.sqrt(1-phi**2)*sigma)
    return np.exp(N + MU) #now N has mean 10,000

def initializeT(numBins, maxGen):
    T = np.random.rand(numBins, maxGen+1)
    return T/T.sum(axis=1)[:, np.newaxis]

def logLike(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2):
    #calculate and return the log likelihood of the complete data


    

    return

def eStep(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2):
    #return updated T1 and T2
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N[:-1])), 0, 0))
    print(sum_log_prob_not_coalesce.shape)
    G = len(N)-1
    #calculate last column of T1 (unnormalized)
    alpha1 = bin_midPoint1/50 #this is a vector
    beta1 = 1-1/(2*N[-1]) #this is just a scalar
    temp1 = 1-beta1*np.exp(-alpha1)
    last_col_1 = sum_log_prob_not_coalesce[-1] + np.log(1-beta1) - alpha1*(1 + G) - np.log(2500) + np.log(G**2/temp1 + (2*G-1)/temp1**2 + 2/temp1**3)

    #calculate last column of T2 (unnormalized)
    alpha2 = bin_midPoint2/50
    beta2 = 1-1/(2*N[-1])
    temp2 = 1-beta2*np.exp(-alpha2)
    last_col_2 = sum_log_prob_not_coalesce[-1] + np.log(1-beta2) - alpha2*(1 + G) - np.log(50) + np.log(G/temp2 + 1/temp2**2)

    #calculate the rest of the column
    log_g_over_50 = np.arange(1, G+1) - np.log(50)
    log_2_times_N_g = np.log(2*N[:-1])
    len_times_g_over_50_1 = bin_midPoint1.reshape((len(bin_midPoint1),1))@(np.arange(1, G+1).reshape((1, G)))
    len_times_g_over_50_2 = bin_midPoint2.reshape((len(bin_midPoint2),1))@(np.arange(1, G+1).reshape((1, G)))

    T1 = 2*log_g_over_50 - len_times_g_over_50_1 - log_2_times_N_g + sum_log_prob_not_coalesce[:-1]
    T2 = log_g_over_50 - len_times_g_over_50_2 - log_2_times_N_g + sum_log_prob_not_coalesce[:-1]

    #this is still log of unnormalized probabilities
    #is normalization necessary?
    T1 = np.append(T1, last_col_1[:,np.newaxis], axis=1)
    T2 = np.append(T2, last_col_2[:, np.newaxis], axis=1)
    normalizing_constant1 = np.apply_along_axis(logsumexp, 1, T1)[:,np.newaxis]
    normalizing_constant2 = np.apply_along_axis(logsumexp, 1, T2)[:,np.newaxis]
    T1 = T1 - normalizing_constant1
    T2 = T2 - normalizing_constant2
    return T1, T2

def mStep(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2):
    #return the updated N estimate, and current loglikelihood
    return



def em(maxGen, bin1, bin2, bin_midPoint1, bin_midPoint2, tol, maxIter):
    N, T1, T2 = initializeN(maxGen), initializeT(bin1.shape[0], maxGen), initializeT(bin2.shape[0], maxGen)

    loglike_prev = logLike(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2)
    T1, T2 = eStep(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2)
    print(np.exp(T1))
    print(np.exp(T2))
    N = mStep(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2)
    loglike_curr = logLike(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2)
    num_Iter = 1

    while (loglike_curr - loglike_prev >= tol and num_iter <= maxIter):
        loglike_prev = loglike_curr
        T1, T2 = eStep(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2)
        N = mStep(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2)
        loglike_curr = logLike(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2)
        num_iter += 1

    if loglike_curr - loglike_prev >= tol:
        print('Warning: EM did not converge. Stopped after {max_Iter} iterations.')

    return N, T1, T2


