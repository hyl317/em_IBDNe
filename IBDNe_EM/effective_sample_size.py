import numpy as np
import gzip
import bisect
import argparse
from scipy.special import logsumexp

L = 3500 #set total genome length to be 3500cM

def expectation_num_segment(N, u):
    G = len(N)
    gen = np.arange(1, G+1)
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))
    log_term1 = logsumexp(np.log(gen/50)-u*gen/50 - np.log(2*N) + sum_log_prob_not_coalesce[:-1])

    N_G = N[-1]
    alpha = np.log(1-1/(2*N_G))-u/50

    log_term2 = -np.log(100*N_G) + sum_log_prob_not_coalesce[-1] + (G+1)*(np.log(2*N_G)-np.log(2*N_G-1)) + alpha*(G+1) + np.log(1+G*(1-np.exp(alpha))) - 2*np.log(1-np.exp(alpha))
    return L*(np.exp(log_term1)+np.exp(log_term2))


def process_ibd_hbd(ibd, hbd, bins, num_haps):
    #read ibd.gz and hbd.gz file
    #return effective sample size for each bin and mean number of IBD segments in each bin

    hapPair2Index = {}
    n_pairs = int(num_haps*(num_haps-1)/2)
    IBD_matrix = np.zeros((n_pairs, len(bins)))
    count = 0

    with gzip.open(ibd, 'rt') as ibd:
        line = ibd.readline()
        while line:
            ind1, hap1, ind2, hap2, chr, start_bp, end_bp, len_cM = line.strip().split('\t')
            chr, start_bp, end_bp, len_cM = int(chr), int(start_bp), int(end_bp), float(len_cM)
            haplotype1 = ind1 + '_' + hap1
            haplotype2 = ind2 + '_' + hap2
            haplotype1, haplotype2 = sorted([haplotype1, haplotype2])
            hapPair = haplotype1 + ':' + haplotype2
            
            tmp = bisect.bisect_left(bins, len_cM)
            pos = tmp if (tmp < len(bins) and bins[tmp] == len_cM) else tmp-1

            if hapPair not in hapPair2Index:
                hapPair2Index[hapPair] = count
                IBD_matrix[count, pos] += 1
                count += 1
            else:
                IBD_matrix[hapPair2Index[hapPair], pos] += 1

            line = ibd.readline()

    with gzip.open(hbd, 'rt') as hbd:
        line = hbd.readline()
        while line:
            ind1, hap1, ind2, hap2, chr, start_bp, end_bp, len_cM = line.strip().split('\t')
            chr, start_bp, end_bp, len_cM = int(chr), int(start_bp), int(end_bp), float(len_cM)
            haplotype1 = ind1 + '_' + hap1
            haplotype2 = ind2 + '_' + hap2
            haplotype1, haplotype2 = sorted([haplotype1, haplotype2])
            hapPair = haplotype1 + ':' + haplotype2
            
            tmp = bisect.bisect_left(bins, len_cM)
            pos = tmp if (tmp < len(bins) and bins[tmp] == len_cM) else tmp-1

            if hapPair not in hapPair2Index:
                hapPair2Index[hapPair] = count
                IBD_matrix[count, pos] += 1
                count += 1
            else:
                IBD_matrix[hapPair2Index[hapPair], pos] += 1

            line = hbd.readline()

    N_REPEAT = 10
    bootstrap_variance = np.zeros(len(bins))
    for j in np.arange(len(bins)):
        print(f'bootstrap for bin {j}')
        bootstrapped_mean = np.zeros(N_REPEAT)
        for rep in np.arange(N_REPEAT):
            samples = IBD_matrix[:,j].flatten()
            resample = np.random.choice(samples, n_pairs, replace=True)
            bootstrapped_mean[rep] = np.mean(resample)
        bootstrap_variance[j] = np.var(bootstrapped_mean)

    #now IBD_matrix is filled
    mean = np.apply_along_axis(np.mean, 0, IBD_matrix)
    return mean/bootstrap_variance, mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ibd', action="store", dest="ibd", type=str, required=True)
    parser.add_argument('--hbd', action="store", dest="hbd", type=str, required=True)
    parser.add_argument('-n', action="store", dest='n', type=int, required=True)
    parser.add_argument('--bins', action="store", dest='bins', type=str, required=False)
    parser.add_argument('-N', action="store", dest="N", type=str, required=False, help="path to file containing reference population size")
    args = parser.parse_args()

    bins = []
    if args.bins == None:
        bins = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        tmp = args.bins.strip().split(',')
        bins = [float(p) for p in tmp]

    #if reference Ne trajectory is provided, calculate the expected number of IBD segments
    if args.N != None:
        N = []
        with open(args.N) as file_N:
            line = file_N.readline()
            while line:
                gen, Ne = line.strip().split('\t')
                Ne = float(Ne)
                N.append(Ne)
                line = file_N.readline()
        N = np.array(N)

        expected_IBD_count = [expectation_num_segment(N, b) for b in bins]

    effective_sample_size, mean_IBD_count = process_ibd_hbd(args.ibd, args.hbd, bins, args.n)
    print(f'mean: {np.flip(np.cumsum(np.flip(mean_IBD_count)))}')
    if args.N != None:
        print(f'expected: {expected_IBD_count}')
    print(f'effective_sample_size: {effective_sample_size}')
    

if __name__ == '__main__':
    main()

