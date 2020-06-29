#main driver script
import argparse
import sys
import numpy as np
import math
from concurrent import futures
from EM_moment_tail import *
from preprocess import *

def main():
    parser = argparse.ArgumentParser(description='IBDNe with Composite Likeilhood and EM.')
    
    parser.add_argument('-i', action="store", dest="ibdSeg", type=str, required=True, 
                        help='Path to .ibd.gz output from Hapibd')
    parser.add_argument('-e', action="store", dest="end", type=str, required=True,
                        help="path to a file containing first and last marker information for every chromosome.")
    parser.add_argument('-n', action="store", dest="numInds", type=int, required=True,
                        help="Number of Individuals in the sample")
    parser.add_argument('--alpha', action="store", dest="alpha", type=float, required=False, default=0.01,
                        help="smoothing parameter. Default: 0.01")
    parser.add_argument('-G', action="store", dest="maxGen", type=int, required=False, default=200, 
                        help='Maximum Number of Generations to Infer. Default: 200')
    parser.add_argument('--tol', action="store", dest="tol", type=float, required=False, default=1e-3,
                        help='Convergence Criterion for EM. Default: 0.001.')
    parser.add_argument('--max_iter', action="store", dest="maxIter", type=int, required=False, default=150,
                        help="Maximum number of iterations for EM. Default: 150")
    parser.add_argument('-o', action="store", dest="out", type=str, required=False, default='ibdne',
                        help="name of output file. Default: ibdne.txt")
    parser.add_argument('--boot', action="store_true", help="Perform bootstrap. Default false.")
    parser.add_argument('-b', action="store", type=int, required=False, default=80,
                    help='number of bootstrap to perform. Default: 80.')
    
    args = parser.parse_args()

    print(f'reading ibd segments from {args.ibdSeg}', flush=True)
    bin1, bin2, bin_midPoint1, bin_midPoint2, chr_len_cM, inds, ibdseg_map1, ibdseg_map2 = processIBD(args.ibdSeg, args.end)
    print(f'A total of {np.sum(bin1) + np.sum(bin2)} IBD segments read for {args.numInds} individuals.', flush=True)
    print(f'Among them, {np.sum(bin2)} reach chromosome end.', flush=True)
    N = em_moment_tail(args.maxGen, bin1, bin2, bin_midPoint1, \
        bin_midPoint2, chr_len_cM, args.numInds, args.alpha, args.tol, args.maxIter)

    print(f'fitted pop size history: {N}')
    
    if not args.boot:
        with open(f'{args.out}.ne.txt','w') as out:
        out.write(f'#{" ".join(sys.argv[1:])}\n')
        for g, ne in enumerate(N):
            out.write(f'{g+1}\t{round(ne, 2)}\n')
    else:
        #start bootstrapping
        
        print('start bootstrapping', flush=True)
        bootstrapped = np.zeros((args.b, args.maxGen))
        with futures.ProcessPoolExecutor() as executor:
            future_list = []
            for i in range(args.b):
                future = executor.submit(bootstrap, inds, ibdseg_map1, ibdseg_map2, \
                    args.maxGen, chr_len_cM, args.numInds, args.alpha, args.tol, args.maxIter, N)
                future_list.append(future)
            done_iter = futures.as_completed(future_list)
            for j, future in enumerate(done_iter):
                try:
                    N_boot = future.result()
                except Exception as e:
                    print(e)
                bootstrapped[j, ] = N_boot

        np.matrix.sort(bootstrapped) #sort by column
        lower_CI_index = math.floor(0.025*args.b)-1
        upper_CI_index = math.floor(0.975*args.b)-1

        with open(f'{args.out}.ne.txt','w') as out:
            out.write(f'#{" ".join(sys.argv[1:])}\n')
            out.write('#generation\tlower_CI\testimate\tupper_CI\n')
            for g, ne in enumerate(N):
                out.write(f'{g+1}\t{round(bootstrapped[lower_CI_index, g])}\t{round(ne, 2)}\t{bootstrapped[upper_CI_index, g]}\n')


if __name__ == '__main__':
    main()

