import argparse
import numpy as np
from exp_num_seg import readNe, expectation_num_segment

chr_len_cM = np.array([286.279234, 268.839622, 223.361095, 214.688476, 204.089357, 192.039918, 
187.220500, 168.003442, 166.359329, 181.144008,
158.218650, 174.679023, 125.706316, 120.202583, 141.860238, 
134.037726, 128.490529, 117.708923, 107.733846, 108.266934, 62.786478, 74.109562])
C = 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', action="store", dest="N", type=str, required=True)
    parser.add_argument('-u', action="store", dest='u', type=float, required=True)
    args = parser.parse_args()
    
    N = readNe(args.N)
    mu = 4*expectation_num_segment(N, args.u)
    


if __name__ == '__main__':
    main()
