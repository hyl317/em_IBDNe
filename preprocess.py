#a few functions for preprocessing input files ready for the EM algorithm
import gzip
import numpy as np
import math

BIN_SIZE = 0.05

def binning(ibdLenList):
    ibdLens = np.array(ibdLenList)
    min, max = np.min(ibdLens), np.max(ibdLens)
    num_bins = math.ceil((max - min)/BIN_SIZE)
    bins = np.zeros(num_bins)
    bin_midPoint = np.zeros(num_bins)

    for i in range(num_bins):
        bin_low, bin_high = min+i*BIN_SIZE, min+(i+1)*BIN_SIZE
        bins[i] = np.count_nonzero((ibdLens >= bin_low) & (ibdLens < bin_high))
        bin_midPoint[i] = (bin_low+bin_high)/2
    bins_trimmed = bins[bins != 0]
    bin_midPoint_trimmed = bin_midPoint[bins != 0]
    print(np.sum(bins_trimmed))
    return bins_trimmed, bin_midPoint_trimmed 

def processIBD(ibd_gz, end_marker):
    endMarker = {}

    with open(end_marker) as end:
        line = end.readline()
        while line:
            chr, rate, cM, phy = line.strip().split('\t')
            chr, phy = int(chr), int(phy)
            if not chr in endMarker:
                endMarker[chr] = [phy]
            else:
                endMarker[chr].append(phy)
            line = end.readline()

    ibdLen1 = [] #store length of IBDs that reside in the middle of a chromosome
    ibdLen2 = [] #store length of IBDs that reaches either end of a chromosome
    with gzip.open(ibd_gz, 'rt') as ibd:
        line = ibd.readline()
        while line:
            ind1, hap1, ind2, hap2, chr, start_bp, end_bp, len_cM = line.strip().split('\t')
            chr, start_bp, end_bp, len_cM = int(chr), int(start_bp), int(end_bp), float(len_cM)
            if start_bp in endMarker[chr] or end_bp in endMarker[chr]:
                ibdLen2.append(len_cM)
            else:
                ibdLen1.append(len_cM)
            line = ibd.readline()

    #calculate bins
    bin1, bin_midPoint1 = binning(ibdLen1)
    bin2, bin_midPoint2 = binning(ibdLen2)
    return bin1, bin2, bin_midPoint1, bin_midPoint2
