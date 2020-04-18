#main driver script
import argparse
import numpy as np
from scipy.special import logsumexp
from EM import *
from EM_moment import *
from EM_moment_tail import *
from plotting import *
from preprocess import *
from EM_by_count import *

def main():
    parser = argparse.ArgumentParser(description='IBDNe with Composite Likeilhood and EM.')
    parser.add_argument('-G', action="store", dest="maxGen", type=int, required=False, default=100, 
                        help='Maximum Number of Generations to Infer')
    parser.add_argument('-i', action="store", dest="ibdSeg", type=str, required=True, 
                        help='Path to .ibd.gz output from Hapibd')
    parser.add_argument('--tol', action="store", dest="tol", type=float, required=False, default=1e-3,
                        help='Convergence Criterion for EM. Otherwise stop after 100 iterations.')
    parser.add_argument('-e', action="store", dest="end", type=str, required=True,
                        help="path to a file containing first and last marker information for every chromosome.")
    parser.add_argument('--alpha', action="store", dest="alpha", type=float, required=False, default=0.01,
                        help="smoothing parameter. Default: 0.01")
    parser.add_argument('--max_iter', action="store", dest="maxIter", type=int, required=False, default=500,
                        help="Maximum number of iterations for EM.")
    parser.add_argument('-o', action="store", dest="out", type=str, required=False, default='ibdne',
                        help="name of output file. If not provided, default is ibdne.txt")
    args = parser.parse_args()

    bin1, bin2, bin_midPoint1, bin_midPoint2, chr_len_cM, numInds = processIBD(args.ibdSeg, args.end)
    #IBD_count_by_bin, bins, total_genome_length, numInds = processIBDandBinning(args.ibdSeg, args.end)
    print(f'A total of {np.sum(bin1)+np.sum(bin2)} IBD segments read for {numInds} individuals.', flush=True)
    print(f'Among them, {np.sum(bin2)} reach chromosome end.', flush=True)
    #N, T1, T2 = em(args.maxGen, bin1, bin2, bin_midPoint1, bin_midPoint2, numInds, args.tol, args.maxIter)
    N, T1, T2 = em_moment_tail(args.maxGen, bin1, bin2, bin_midPoint1, bin_midPoint2, chr_len_cM, numInds, args.alpha, args.tol, args.maxIter)
    #em_by_count(IBD_count_by_bin, bins, total_genome_length, numInds, args.maxGen, args.alpha, args.maxIter, args.tol)

    with open(f'{args.out}.ne.txt','w') as out:
        for g, ne in enumerate(N):
            out.write(f'{g+1}\t{round(ne, 2)}\n')

    cutoff = 7
    with open(f'{args.out}.tmrca.txt','w') as out_tmrca:
        out_tmrca.write(f'#median_bin_length(cM)\tmean_posterior_tmrca\tprobability_coalesce_within_{cutoff}_generation\n')
        tmp = np.log(np.arange(1, args.maxGen+1)) + T1[:, :-1]
        posterior_expectation_tmrca = np.exp(np.apply_along_axis(logsumexp, 1, tmp))
        prob_coalesce_within_cutoff = np.exp(np.apply_along_axis(logsumexp, 1, T1[:,:cutoff]))
        for bin_median, tmrca, prob in zip(bin_midPoint1, posterior_expectation_tmrca, prob_coalesce_within_cutoff):
            out_tmrca.write(f'{round(bin_median,3)}\t{round(tmrca,2)}\t{prob}\n')


if __name__ == '__main__':
    main()

