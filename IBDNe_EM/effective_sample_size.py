import numpy as np
import gzip
import bisect
import argparse


def process_ibd_hbd(ibd, hbd, bins, num_haps):
    #read ibd.gz and hbd.gz file
    #return effective sample size for each bin and mean number of IBD segments in each bin

    hapPair2Index = {}
    n_pairs = num_haps*(num_haps-1)/2
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
            
            tmp = bisect.bisect_left(bin, len_cM)
            pos = tmp if (tmp < len(bin) and bin[tmp] == len_cM) else tmp-1

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
            
            tmp = bisect.bisect_left(bin, len_cM)
            pos = tmp if (tmp < len(bin) and bin[tmp] == len_cM) else tmp-1

            if hapPair not in hapPair2Index:
                hapPair2Index[hapPair] = count
                IBD_matrix[count, pos] += 1
                count += 1
            else:
                IBD_matrix[hapPair2Index[hapPair], pos] += 1

            line = hbd.readline()

    N_REPEAT = 10000
    bootstrap_variance = np.zeros(len(bins))
    for j in np.arange(len(bins)):
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
    args = parser.parse_args()

    bin = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    effective_sample_size, mean_IBD_count = process_ibd_hbd(args.ibd, args.hbd, bin, args.n)
    print(f'effective_sample_size={effective_sample_size}')
    print(f'mean={np.flip(np.cumsum(np.flip(mean_IBD_count)))}')

if __name__ == '__main__':
    main()

