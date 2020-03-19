#main driver script
import argparse
import numpy as np
from EM import *
from EMbyMoment import *
from plotting import *
from preprocess import *

def main():
    parser = argparse.ArgumentParser(description='IBDNe with Composite Likeilhood and EM.')
    parser.add_argument('-G', action="store", dest="maxGen", type=int, required=False, default=100, 
                        help='Maximum Number of Generations to Infer')
    parser.add_argument('-i', action="store", dest="ibdSeg", type=str, required=True, 
                        help='Path to .ibd.gz output from Hapibd')
    parser.add_argument('--tol', action="store", dest="tol", type=float, required=False, default=1e-6,
                        help='Convergence Criterion for EM. Otherwise stop after 100 iterations.')
    parser.add_argument('-e', action="store", dest="end", type=str, required=True,
                        help="path to a file containing first and last marker information for every chromosome.")
    parser.add_argument('--max_iter', action="store", dest="maxIter", type=int, required=False, default=100,
                        help="Maximum number of iterations for EM.")
    args = parser.parse_args()

    bin1, bin2, bin_midPoint1, bin_midPoint2, chr_len_cM, numInds = processIBD(args.ibdSeg, args.end)
    print(f'A total of {np.sum(bin1)+np.sum(bin2)} IBD segments read for {numInds} individuals.', flush=True)
    print(f'Among them, {np.sum(bin2)} reach chromosome end.', flush=True)
    N, T1, T2 = em(args.maxGen, bin1, bin2, bin_midPoint1, bin_midPoint2, numInds, args.tol, args.maxIter)
    #N, T1, T2 = em_byMoment(args.maxGen, bin1, bin2, bin_midPoint1, bin_midPoint2, chr_len_cM, numInds, args.tol, args.maxIter)
    
    with open('ibdne.txt','w') as out:
        for g, ne in enumerate(N):
            out.write(f'{g+1}\t{ne}\n')


if __name__ == '__main__':
    main()

