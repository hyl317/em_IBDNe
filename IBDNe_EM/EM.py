#primary functions for EM algorithm
import numpy as np
import math
import statsmodels.api as sm
from scipy.special import logsumexp
from scipy.optimize import minimize
from plotting import *
from misc import *



def logLike(N, bin1, bin2, bin_midPoint1, bin_midPoint2, N_p, alpha):
    ##calculate and return the log likelihood of the complete data
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))
    G = len(N)
    ##for IBD segments in the middle of a chromosome, calculate the prob of coalescing earlier than G generations in the past
    alpha1 = bin_midPoint1/50 #this is a vector
    beta1 = 1-1/(2*N[-1]) #this is just a scalar
    temp1 = 1-beta1*np.exp(-alpha1)
    last_col_1 = sum_log_prob_not_coalesce[-1] + np.log(1-beta1) - alpha1*(1 + G) - np.log(2500) + np.log(G**2/temp1 + (2*G-1)/temp1**2 + 2/temp1**3)

    ###for IBD segments that reach either end of a chromosome, calculate the prob of coalescing earlier than G generations in the past
    alpha2 = bin_midPoint2/50
    beta2 = 1-1/(2*N[-1])
    temp2 = 1-beta2*np.exp(-alpha2)
    last_col_2 = sum_log_prob_not_coalesce[-1] + np.log(1-beta2) - alpha2*(1 + G) - np.log(50) + np.log(G/temp2 + 1/temp2**2)

    ##calculate, for each bin, the IBD segments coalesce at 1,2,...,G generations in the past
    log_g_over_50 = np.log(np.arange(1, G+1)/50)
    log_2_times_N_g = np.log(2*N)
    len_times_g_over_50_1 = bin_midPoint1.reshape((len(bin_midPoint1),1))@(np.arange(1, G+1).reshape((1, G)))/50
    len_times_g_over_50_2 = bin_midPoint2.reshape((len(bin_midPoint2),1))@(np.arange(1, G+1).reshape((1, G)))/50

    T1_temp = 2*log_g_over_50 - len_times_g_over_50_1 - log_2_times_N_g + sum_log_prob_not_coalesce[:-1]
    T2_temp = log_g_over_50 - len_times_g_over_50_2 - log_2_times_N_g + sum_log_prob_not_coalesce[:-1]
    T1_unnormalized = np.append(T1_temp, last_col_1[:,np.newaxis], axis=1)
    T2_unnormalized = np.append(T2_temp, last_col_2[:,np.newaxis], axis=1)

    ## add penalty term to the log likelihood
    N_shifted = np.roll(N,-1)
    N_shifted[-1] = N[-1]
    diff = N_shifted - N
    penalty = alpha*np.sum(np.dot(diff, diff))/N_p
    print(T1_unnormalized)
    print(T2_unnormalized)

    return np.sum(bin1*np.apply_along_axis(logsumexp, 1, T1_unnormalized)) + np.sum(bin2*np.apply_along_axis(logsumexp, 1, T2_unnormalized)) - penalty

def eStep(N, bin1, bin2, bin_midPoint1, bin_midPoint2):
    #return updated T1 and T2
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))
    #print(sum_log_prob_not_coalesce.shape)
    G = len(N)
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
    log_g_over_50 = np.log(np.arange(1, G+1)/50)
    log_2_times_N_g = np.log(2*N)
    len_times_g_over_50_1 = bin_midPoint1.reshape((len(bin_midPoint1),1))@(np.arange(1, G+1).reshape((1, G)))/50
    len_times_g_over_50_2 = bin_midPoint2.reshape((len(bin_midPoint2),1))@(np.arange(1, G+1).reshape((1, G)))/50

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

def negCompleteDataLikelihood(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, N_p, alpha):
    ## add penalty term to the log likelihood
    N_shifted = np.roll(N,-1)
    N_shifted[-1] = N[-1]
    diff = N_shifted - N
    penalty = (alpha*np.sum(np.dot(diff, diff)))/N_p

    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0)) #this has length maxGen+1

    #for IBD segments in the middle of the chromosome
    G = len(N)
    log_g_over_50 = np.log(np.arange(1, G+1)/50)
    log_2_times_N_g = np.log(2*N)
    len_times_g_over_50_1 = bin_midPoint1.reshape((len(bin_midPoint1),1))@(np.arange(1, G+1).reshape((1, G)))/50
    len_times_g_over_50_2 = bin_midPoint2.reshape((len(bin_midPoint2),1))@(np.arange(1, G+1).reshape((1, G)))/50

    matrix1 = 2*log_g_over_50 - len_times_g_over_50_1 + sum_log_prob_not_coalesce[:-1] - log_2_times_N_g
    matrix2 = log_g_over_50 - len_times_g_over_50_2 + sum_log_prob_not_coalesce[:-1] - log_2_times_N_g

    #calculate last column of matrix1
    alpha1 = bin_midPoint1/50 #this is a vector
    beta1 = 1-1/(2*N[-1]) #this is just a scalar
    temp1 = 1-beta1*np.exp(-alpha1)
    last_col_1 = sum_log_prob_not_coalesce[-1] + np.log(1-beta1) - alpha1*(1 + G) - np.log(2500) + np.log(G**2/temp1 + (2*G-1)/temp1**2 + 2/temp1**3)

    #calculate last column of matrix2
    alpha2 = bin_midPoint2/50
    beta2 = 1-1/(2*N[-1])
    temp2 = 1-beta2*np.exp(-alpha2)
    last_col_2 = sum_log_prob_not_coalesce[-1] + np.log(1-beta2) - alpha2*(1 + G) - np.log(50) + np.log(G/temp2 + 1/temp2**2)

    matrix1 = np.append(matrix1, last_col_1[:,np.newaxis], axis=1)
    matrix2 = np.append(matrix2, last_col_2[:,np.newaxis], axis=1)
    term1 = np.sum(np.exp(T1 + np.log(bin1)[:,np.newaxis])*matrix1)
    term2 = np.sum(np.exp(T2 + np.log(bin2)[:,np.newaxis])*matrix2)
    #print(f'term1:{term1}',flush=True)
    #print(f'term2:{term2}', flush=True)
    #print(f'penalty:{penalty}', flush=True)
    #if math.isnan(term1) or math.isnan(term2):
    #print(N)
    return -term1-term2-penalty


def jacobian(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, N_p, alpha):
    maxGen = len(N)
    T1 = np.log(bin1)[:,np.newaxis] + T1
    T2 = np.log(bin2)[:,np.newaxis] + T2
    sum_over_every_column1 = np.apply_along_axis(logsumexp, 0, T1)
    sum_over_every_column2 = np.apply_along_axis(logsumexp, 0, T2)
    logA = np.logaddexp(sum_over_every_column1, sum_over_every_column2)[:-1]

    temp1 = np.logaddexp.accumulate(np.fliplr(sum_over_every_column1.reshape(1, maxGen+1)).flatten())
    cum_sum_to_the_right1 = np.fliplr(temp1.reshape(1, len(temp1))).flatten()[1:]
    temp2 = np.logaddexp.accumulate(np.fliplr(sum_over_every_column2.reshape(1, maxGen+1)).flatten())
    cum_sum_to_the_right2 = np.fliplr(temp2.reshape(1, len(temp2))).flatten()[1:]
    logB = np.logaddexp(cum_sum_to_the_right1, cum_sum_to_the_right2)

    #likelihood_term = np.log(N) - logA - np.log(N*(2*N-1)) + logB
    likelihood_term = -np.exp(logA)/N + np.exp(logB)/(N*(2*N-1))
    N_left = np.roll(N,-1)
    N_right = np.roll(N,1)
    penalty_term = 4*N - 2*(N_left + N_right)
    penalty_term[0] = 2*(N[0] - N[1])
    penalty_term[-1] = 2*(N[-1] - N[-2])

    #print(f'logA:{logA}')
    #print(f'logB:{logB}')
    #print(f'penalty component in gradient: {alpha*penalty_term/N_p}')
    #print(f'likelihood component in gradient: {likelihood_term}')

    #return -np.exp(likelihood_term) - alpha*penalty_term/N_p
    print(f'calculated jacobian is {-likelihood_term - alpha*penalty_term/N_p}')
    return -likelihood_term - alpha*penalty_term/N_p


def mStep(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, N_p, alpha):
    bnds = [(0, np.inf) for n in N]
    result = minimize(negCompleteDataLikelihood, N, args=(T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, N_p, alpha), 
                      method='Nelder-Mead', tol=1e-6, bounds=bnds)
    ##return the updated N estimate
    #maxGen = len(N)
    #N_updated = np.zeros(maxGen)
    ##calculate N through 1,2,..., G-1
    #T1 = np.log(bin1)[:,np.newaxis] + T1
    #T2 = np.log(bin2)[:,np.newaxis] + T2
    #sum_over_every_column1 = np.apply_along_axis(logsumexp, 0, T1)
    #sum_over_every_column2 = np.apply_along_axis(logsumexp, 0, T2)
    #logA = np.logaddexp(sum_over_every_column1, sum_over_every_column2)[:-2]
    #temp1 = np.logaddexp.accumulate(np.fliplr(sum_over_every_column1.reshape(1, maxGen+1)).flatten())
    #cum_sum_to_the_right1 = np.fliplr(temp1.reshape(1, len(temp1))).flatten()[1:-1]
    #temp2 = np.logaddexp.accumulate(np.fliplr(sum_over_every_column2.reshape(1, maxGen+1)).flatten())
    #cum_sum_to_the_right2 = np.fliplr(temp2.reshape(1, len(temp2))).flatten()[1:-1]
    #logB = np.logaddexp(cum_sum_to_the_right1, cum_sum_to_the_right2)

    #N_updated[:maxGen-1] = (1+np.exp(logB-logA))/2
    #N_updated[maxGen-1] = N_updated[maxGen-2]
    print(f'N updated:{result}', flush=True)
    return result.x


def test_loglike(maxGen, bin1, bin2, bin_midPoint1, bin_midPoint2, N_p, alpha):
    refNe = refFinNe()
    randomNe = initializeN_autoreg(maxGen)
    refLog = logLike(refNe, bin1, bin2, bin_midPoint1, bin_midPoint2, N_p, alpha)
    randomLog = logLike(randomNe, bin1, bin2, bin_midPoint1, bin_midPoint2, N_p, alpha)
    print(f'reference Ne: {refNe}')
    print(f'random Ne: {randomNe}')
    print(f'penalized log likelihood of reference Ne trajectory: {refLog}')
    print(f'penalized log likelihood of random Ne trajectory: {randomLog}')

def test_loglike_toyEx():
    bin1 = np.array([1,1,1])
    bin_midPoint1 = np.array([2,4,6])
    bin2 = np.array([1,1,1])
    bin_midPoint2 = np.array([2,4,6])
    N = initializeN_Uniform(5, 1000)
    N_p, alpha = 20, 0.05
    logLike(N, bin1, bin2, bin_midPoint1, bin_midPoint2, N_p, alpha)



def em(maxGen, bin1, bin2, bin_midPoint1, bin_midPoint2, numInds, tol, maxIter):
    alpha = 0.05
    N_p = 2*numInds*(2*numInds-2)/2
    test_loglike_toyEx()
    test_loglike(maxGen, bin1, bin2, bin_midPoint1, bin_midPoint2, N_p, alpha)
    N, T1, T2 = initializeN_Uniform(maxGen, 10000), initializeT_Random(bin1.shape[0], maxGen), initializeT_Random(bin2.shape[0], maxGen)
    print(f"initial N:{N}")
    loglike_prev = logLike(N, bin1, bin2, bin_midPoint1, bin_midPoint2, N_p, alpha)
    T1, T2 = eStep(N, bin1, bin2, bin_midPoint1, bin_midPoint2)
    N = mStep(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, N_p, alpha)
    loglike_curr = logLike(N, bin1, bin2, bin_midPoint1, bin_midPoint2, N_p, alpha)
    num_iter = 1
    plotPosterior(np.exp(T1.T), bin_midPoint1, np.arange(1, maxGen+2), title=f'Posterior Distribution for Iteration {num_iter}')
    while (loglike_curr - loglike_prev >= tol and num_iter < maxIter):
        print(f'iteration{num_iter} done. Likelihood improved by {loglike_curr-loglike_prev}')
        loglike_prev = loglike_curr
        T1, T2 = eStep(N, bin1, bin2, bin_midPoint1, bin_midPoint2)
        N = mStep(N, T1, T2, bin1, bin2, bin_midPoint1, bin_midPoint2, N_p, alpha)
        loglike_curr = logLike(N, bin1, bin2, bin_midPoint1, bin_midPoint2, N_p, alpha)
        num_iter += 1
    
    print(f'iteration{num_iter} done. Likelihood improved by {loglike_curr-loglike_prev}')
    plotPosterior(np.exp(T1.T), bin_midPoint1, np.arange(1, maxGen+2), title=f'Posterior Distribution for Iteration {num_iter}')    
    print(N)
    if loglike_curr - loglike_prev >= tol:
        print(f'Warning: EM did not converge. Stopped after {maxIter} iterations.')
    return N, T1, T2


