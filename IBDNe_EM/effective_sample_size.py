import numpy as np
import gzip
import bisect
import argparse
from scipy.special import logsumexp
from misc import initializeN_autoreg
from scipy.ndimage.interpolation import shift
from scipy.optimize import minimize

def expectation_num_segment(N, u, total_genome_length):
    G = len(N)
    gen = np.arange(1, G+1)
    sum_log_prob_not_coalesce = np.cumsum(np.insert(np.log(1-1/(2*N)), 0, 0))
    log_term1 = logsumexp(np.log(gen/50)-u*gen/50 - np.log(2*N) + sum_log_prob_not_coalesce[:-1])
    
    N_G = N[-1]
    alpha = np.log(1-1/(2*N_G))-u/50

    log_term2 = -np.log(100*N_G) + sum_log_prob_not_coalesce[-1] + (G+1)*(np.log(2*N_G)-np.log(2*N_G-1)) + alpha*(G+1) + np.log(1+G*(1-np.exp(alpha))) - 2*np.log(1-np.exp(alpha))
    return total_genome_length*(np.exp(log_term1)+np.exp(log_term2))

#probably need to add a regularization term
def neg_loglikelihood(N, N_eff, mean_IBD_count, total_genome_length, bins, alpha):
    cumulative_expected_IBD_count = [expectation_num_segment(N, b, total_genome_length) for b in bins]
    expected_IBD_count = cumulative_expected_IBD_count - shift(cumulative_expected_IBD_count, -1, cval=0)
    loglike = np.sum(N_eff*(mean_IBD_count*np.log(expected_IBD_count)-expected_IBD_count))
    penalty = alpha*np.sum(np.diff(N, n=2)**2)
    return -loglike + penalty

def fit_mle_N(N_eff, mean_IBD_count, total_genome_length, bins, G, alpha):
    init_N = initializeN_autoreg(G)
    print(f'N_eff: {N_eff}', flush=True)
    print(f'value of obj function at random start: {neg_loglikelihood(init_N, N_eff, mean_IBD_count, total_genome_length, bins, alpha)}', flush=True)
    result = minimize(neg_loglikelihood, init_N, args=(N_eff, mean_IBD_count, total_genome_length, bins, alpha), method='Powell', options={'maxfev':1e7})
    print(result)
    return result.x

def process_ibd_hbd(ibd, hbd, endMarkers, bins, num_haps):
    #read ibd.gz and hbd.gz file
    #return effective sample size for each bin and mean number of IBD segments in each bin

    hapPair2Index = {}
    n_pairs = int(num_haps*(num_haps-1)/2)
    IBD_matrix = np.zeros((n_pairs, len(bins)))
    IBD_count_censored = np.zeros(len(bins))
    count = 0

    #read in end markers
    endMarker_bp = {}
    endMarker_cM = {}

    with open(endMarkers) as end:
        line = end.readline()
        while line:
            chr, rate, cM, phy = line.strip().split('\t')
            chr, cM, phy = int(chr), float(cM), int(phy)
            if not chr in endMarker_bp:
                endMarker_bp[chr] = [phy]
                endMarker_cM[chr] = [cM]
            else:
                endMarker_bp[chr].append(phy)
                endMarker_cM[chr].append(cM)
            line = end.readline()

    total_genome_length = 0
    for chr, start_end in endMarker_cM.items():
        total_genome_length += abs(start_end[1]-start_end[0])

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

            if start_bp in endMarker_bp[chr] or end_bp in endMarker_bp[chr]:
                IBD_count_censored[pos] += 1

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

            if start_bp in endMarker_bp[chr] or end_bp in endMarker_bp[chr]:
                IBD_count_censored[pos] += 1

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
        print(f'bootstrap for bin {j}', flush=True)
        bootstrapped_mean = np.zeros(N_REPEAT)
        for rep in np.arange(N_REPEAT):
            samples = IBD_matrix[:,j].flatten()
            resample = np.random.choice(samples, n_pairs, replace=True)
            bootstrapped_mean[rep] = np.mean(resample)
        bootstrap_variance[j] = np.var(bootstrapped_mean)

    #now IBD_matrix is filled
    count_total_IBD = np.apply_along_axis(np.sum, 0, IBD_matrix)
    count_uncensored_IBD = count_total_IBD - IBD_count_censored
    tmp = np.copy(count_uncensored_IBD)

    #redistribute censored IBD into bins
    frac = count_total_IBD/np.sum(count_total_IBD)
    #print(f'frac is {frac}')
    for b, count in enumerate(IBD_count_censored):
        #print(f'count is {count}, redistributed count is {(count*frac[b:])/np.sum(frac[b:])}')
        tmp[b:] += (count*frac[b:])/np.sum(frac[b:])

    mean = tmp/n_pairs

    return mean/bootstrap_variance, mean, total_genome_length

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ibd', action="store", dest="ibd", type=str, required=True)
    parser.add_argument('--hbd', action="store", dest="hbd", type=str, required=True)
    parser.add_argument('-e', action='store', dest='end', type=str, required=True, help="path to files of end markers")
    parser.add_argument('-n', action="store", dest='n', type=int, required=True, help="number of haplotypes")
    parser.add_argument('-G', action='store', dest='G', type=int, required=False, default=200, help='maximum number of generations to infer')
    parser.add_argument('--alpha', action='store', dest='alpha', type=float, required=False, default=0.01, help='alpha')
    parser.add_argument('--bins', action="store", dest='bins', type=str, required=False)
    #parser.add_argument('-N', action="store", dest="N", type=str, required=False, help="path to file containing reference population size")
    args = parser.parse_args()

    bins = []
    if args.bins == None:
        bins = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        tmp = args.bins.strip().split(',')
        bins = [float(p) for p in tmp]

    effective_sample_size, mean_IBD_count, total_genome_length = process_ibd_hbd(args.ibd, args.hbd, args.end, bins, args.n)
    N = fit_mle_N(effective_sample_size, mean_IBD_count, total_genome_length, bins, args.G, args.alpha)
    print(f'mle fitted Ne trajectory: {N}', flush=True)
    #calculate expected number of IBD segments
    #if reference Ne trajectory is provided, calculate the expected number of IBD segments
    #if args.N != None:
    #    N = []
    #    with open(args.N) as file_N:
    #        line = file_N.readline()
    #        while line:
    #            gen, Ne = line.strip().split('\t')
    #            Ne = float(Ne)
    #            N.append(Ne)
    #            line = file_N.readline()
    #    N = np.array(N)

    #    expected_IBD_count = [expectation_num_segment(N, b, total_genome_length) for b in bins]


    #print(f'mean: {np.flip(np.cumsum(np.flip(mean_IBD_count)))}')
    #if args.N != None:
    #    print(f'expected: {expected_IBD_count}')
    

if __name__ == '__main__':
    main()

