#main driver script
import argparse
from EM import *
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
    args = parser.parse_args()

    bin1, bin2, bin_midPoint1, bin_midPoint2 = processIBD(args.ibdSeg, args.end)
    print(bin1)
    print(bin2)
    print(bin_midPoint1)
    print(bin_midPoint2)

if __name__ == '__main__':
    main()

