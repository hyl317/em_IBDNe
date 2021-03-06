#approximating infinite sums

from scipy.integrate import quad
from scipy.special import logsumexp
import numpy as np
import argparse

chr_len_cM = np.array([286.279234, 268.839622, 223.361095, 214.688476, 204.089357, 192.039918, 
187.220500, 168.003442, 166.359329, 181.144008,
158.218650, 174.679023, 125.706316, 120.202583, 141.860238, 
134.037726, 128.490529, 117.708923, 107.733846, 108.266934, 62.786478, 74.109562])
C = 2

L = np.sum(chr_len_cM)

#maxGen = 100
#Ne = 2e4

#N = np.full(maxGen, Ne)
#chromLen = np.full(30, 100) #suppose 30 chromosomes of length 100Mb each
#n = 1000
#n_p = (2*n*(2*n-2))/2
#N_past = N[-1]
#CONSTANT = np.exp(np.log(n_p) - np.log(2*N_past) + np.sum(np.log(1-1/(2*N))))
#C = 2


def readNe(NeFile):
    N = []
    with open(NeFile) as Ne:
        line = Ne.readline()
        while line:
            g, ne = line.strip().split('\t')
            N.append(float(ne))
            line = Ne.readline()
    return np.array(N)

#def log_partB(g, N_g, maxGen, C, chromLen):
#    part3 = np.sum((C*g/50 + 1)*chromLen) - len(chromLen)*(C**2)*g/50
#    part2 = -C*g/50
#    part1 = (g-maxGen-1)*np.log(1-1/(2*N_g))
#    return np.exp(part1 + part2 + np.log(part3))


#integral1, err2 = quad(log_partB, maxGen+1, np.inf, args=(N_past, maxGen, C, chromLen))
#integral2, err2 = quad(log_partB, maxGen, np.inf, args=(N_past, maxGen, C, chromLen))
#print(integral1)
#print(integral2)

#L = 3500 #set total genome length to be 3500cM


def expectation_num_segment(N, u):
    G = len(N)
    gen = np.arange(1, G+1)
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))
    log_term1 = logsumexp(np.log(gen/50)-u*gen/50 - np.log(2*N) + sum_log_prob_not_coalesce[:-1])

    N_G = N[-1]
    alpha = np.log(1-1/(2*N_G))-u/50

    log_term2 = -np.log(100*N_G) + sum_log_prob_not_coalesce[-1] + \
        (G+1)*(np.log(2*N_G)-np.log(2*N_G-1)) + alpha*(G+1) + np.log(1+G*(1-np.exp(alpha))) -\
       2*np.log(1-np.exp(alpha))

    return L*(np.exp(log_term1)+np.exp(log_term2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', action="store", dest="N", type=str, required=True)
    parser.add_argument('-u', action="store", dest='u', type=float, required=True)
    args = parser.parse_args()
    
    N = readNe(args.N)
    expected_num_seg = expectation_num_segment(N, args.u)
    print(f'expected number of IBD segments longer than {args.u}cM for pairs of haplotypes: {expected_num_seg}')

if __name__ == '__main__':
    main()


#def log_expectedIBD_beyond_maxGen_given_Ne(N, chr_len_cM, maxGen, n_p):
#    def partB(g, N_g, maxGen, C, chromLen):
#        part3 = np.sum((C*g/50 + 1)*chromLen) - len(chromLen)*(C**2)*g/50
#        part2 = -C*g/50
#        part1 = (g-maxGen-1)*np.log(1-1/(2*N_g))
#        return np.exp(part1 + part2 + np.log(part3))
#    N_past = N[-1]
#    integral1, err1 = quad(partB, maxGen+1, np.inf, args=(N_past, maxGen, C, chr_len_cM))
#    integral2, err2 = quad(partB, maxGen, np.inf, args=(N_past, maxGen, C, chr_len_cM))
#    #print(f'N={N}')
#    #print(f'evaluated at N_g={N_past} and the integral is {integral}')
#    return np.log(n_p) - np.log(2*N_past) + np.sum(np.log(1-1/(2*N))) + np.log((integral1 + integral2)/2)

#def expectedIBDSharing(N, chr_len_cM):
#    G = len(N)
#    gen = np.arange(1, G+1)
#    log_term3 = np.log(np.sum(C*(chr_len_cM[:,np.newaxis]@gen.reshape((1, G)))/50 + chr_len_cM[:,np.newaxis] - ((C**2)*gen)/50, axis=0))
#    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))[:-1]
#    log_expectation = np.log(4) + sum_log_prob_not_coalesce - np.log(2*N) - C*gen/50 + log_term3
#    log_expectation = np.append(log_expectation, log_expectedIBD_beyond_maxGen_given_Ne(N, chr_len_cM, G, 4)) #need to calculate expected amount of IBD coalescing beyond maxGen generations into the past
#    return np.sum(np.exp(log_expectation))

#def main():
#    parser = argparse.ArgumentParser()
#    parser.add_argument('-N', action="store", dest="N", type=str, required=False)
#    args = parser.parse_args()

#    N = readNe(args.N)
#    backgroundIBD = expectedIBDSharing(N, chr_len_cM)
#    print(f'total genome length{np.sum(chr_len_cM)}')
#    print(f'expected backgroundIBD sharing amount: {round(backgroundIBD,2)}')

#if __name__ == '__main__':
#    main()
