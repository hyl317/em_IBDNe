#approximating infinite sums

from scipy.integrate import quad
from scipy.special import logsumexp
import numpy as np
import argparse


maxGen = 100
Ne = 2e4

N = np.full(maxGen, Ne)
chromLen = np.full(30, 100) #suppose 30 chromosomes of length 100Mb each
n = 1000
n_p = (2*n*(2*n-2))/2
N_past = N[-1]
CONSTANT = np.exp(np.log(n_p) - np.log(2*N_past) + np.sum(np.log(1-1/(2*N))))
C = 2
def log_partB(g, N_g, maxGen, C, chromLen):
    part3 = np.sum((C*g/50 + 1)*chromLen) - len(chromLen)*(C**2)*g/50
    part2 = -C*g/50
    part1 = (g-maxGen-1)*np.log(1-1/(2*N_g))
    return np.exp(part1 + part2 + np.log(part3))


integral1, err2 = quad(log_partB, maxGen+1, np.inf, args=(N_past, maxGen, C, chromLen))
integral2, err2 = quad(log_partB, maxGen, np.inf, args=(N_past, maxGen, C, chromLen))
print(integral1)
print(integral2)

#L = 3500 #set total genome length to be 3500cM


#def expectation_num_segment(N, u):
#    G = len(N)
#    gen = np.arange(1, G+1)
#    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))
#    log_term1 = logsumexp(np.log(gen/50)-u*gen/50 - np.log(2*N) + sum_log_prob_not_coalesce[:-1])

#    N_G = N[-1]
#    alpha = np.log(1-1/(2*N_G))-u/50

#    log_term2 = -np.log(100*N_G) + sum_log_prob_not_coalesce[-1] + (G+1)*(np.log(2*N_G)-np.log(2*N_G-1)) + alpha*(G+1) + np.log(1+G*(1-np.exp(alpha))) - 2*np.log(1-np.exp(alpha))
#    return L*(np.exp(log_term1)+np.exp(log_term2))


#def main():
#    parser = argparse.ArgumentParser()
#    parser.add_argument('-N', action="store", dest="N", type=str, required=False)
#    args = parser.parse_args()
    
#    N = []
#    if args.N == None:
#        N = np.full(100, 20000) #constant effective population size of 2e4
#    else:
#        with open(args.N) as file_N:
#            line = file_N.readline()
#            while line:
#                gen, Ne = line.strip().split('\t')
#                Ne = float(Ne)
#                N.append(Ne)
#                line = file_N.readline()
#        N = np.array(N)

#    #print(N)
#    for u in np.arange(2, 11):
#        expected_num_seg = expectation_num_segment(N, u)
#        print(f'expected number of IBD segments longer than {u}cM for pairs of haplotypes: {expected_num_seg}')

#if __name__ == '__main__':
#    main()
