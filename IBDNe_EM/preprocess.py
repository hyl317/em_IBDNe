#a few functions for preprocessing input files ready for the EM algorithm
import gzip
import numpy as np
import math
import bisect
import re
import sys
from collections import defaultdict
from collections import namedtuple

BIN_SIZE = 0.05
STEP = 0.25

def calcCM(bp1, bp2, cM1, cM2, query_bp):
    assert query_bp > bp1
    assert query_bp < bp2
    return cM1 + (cM2-cM1)*((query_bp-bp1)/(bp2-bp1))

def calcBP(bp1, bp2, cM1, cM2, query_cM):
    assert query_cM > cM1
    assert query_cM < cM2
    return int(bp1 + (bp2 - bp1)*((query_cM - cM1)/(cM2 - cM1)))


class genetic_map(object):
    def init(self, mapFile):
        self.dict_bp = defaultdict(lambda : [])
        self.dict_cM = defaultdict(lambda : [])
        with open(mapFile) as map:
            line = map.readline()
            while line:
                chr, bp, _, cM = line.strip().split('\t')
                bp, cM = int(bp), float(cM)
                dict_bp[chr].append(bp)
                dict_cM[chr].append(cM)
                line = map.readline()
        
        self.total_marker_num = {chr:len(self.dict_bp[chr]) for chr in self.dict_bp.keys()}


    def calcLengthCM(self, chr, start_bp, end_bp):
        return self.getCM(chr, end_bp) - self.getCM(chr, start_bp)

    def getBP(self, chr, cM):
        tmp = bisect.bisect_left(self.dict_cM[chr], cM)

        if tmp == 0 or tmp == total_marker_num[chr]:
            print('segment exceeds boundary of the provided genetic map')
            sys.exit()

        bp = self.dict_bp[chr][tmp] if sefl.dcit_cM[chr][tmp] == cM else \
            calcBP(self.dict_bp[chr][tmp-1], self.dict_bp[chr][tmp],
            self.dict_cM[chr][tmp-1], self.dict_cM[chr][tmp], cM)
        return bp

    def getCM(self, chr, bp):
        tmp = bisect.bisect_left(self.dict_bp[chr][bp], bp)

        if tmp == 0 or tmp == total_marker_num[chr]:
            print('segment exceeds boundary of the provided genetic map')
            sys.exit()

        cM = self.dict_cM[chr][tmp] if bp == self.dict_bp[chr][tmp] \
            else calcCM(self.dict_bp[chr][tmp-1], self.dict_bp[chr][tmp],
            self.dict_cM[chr][tmp-1], self.dict_cM[chr][tmp], bp)
        return cM

    def binByStep(self, step, endMarker_cM):
        breakpoints = defaultdict(lambda : [])
        tail = {}
        for chr in self.dict_bp.keys():
            START, END = min(endMarker_cM[chr]), max(endMarker_cM[chr])
            interval_start = START
            interval_end = interval_start + step
            breakpoints[chr].append(self.getBP(chr, START))
            while interval_end < END:
                tmp = bisect.bisect_left(self.dict_cM[chr], interval_end)
                bp = self.dict_bp[chr][tmp] if sefl.dcit_cM[chr][tmp] == interval_end else \
                    calcBP(self.dict_bp[chr][tmp-1], self.dict_bp[chr][tmp],
                    self.dict_cM[chr][tmp-1], self.dict_cM[chr][tmp], interval_end)
                breakpoints[chr].append(bp)
                interval_start = interval_end
                interval_end += step
            tail[chr] = max(endMarker_cM[chr]) - (interval_end - step)
        
        #note this ignores the last bin
        return breakpoints

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
    return bins_trimmed, bin_midPoint_trimmed 

def processIBD(ibd_gz, end_marker, mapFile=None):
    endMarker_bp = defaultdict(lambda : [])
    endMarker_cM = defaultdict(lambda : [])

    with open(end_marker) as end:
        line = end.readline()
        while line:
            chr, rate, cM, phy = line.strip().split('\t')
            chr, cM, phy = int(chr), float(cM), int(phy)
            endMarker_bp[chr].append(phy)
            endMarker_cM[chr].append(cM)
            line = end.readline()
    
    # IBD = namedtuple("chr start_bp end_bp")
    # IBDset = set()
    # with gzip.open(ibd_gz, 'rt') as ibd:
    #     line = ibd.readline()
    #     while line:
    #         _, _, _, _, chr, bp1, bp2, _ = line.strip().split('\t')
    #         IBDset.add(IBD(chr, int(bp1), int(bp2)))
    #         line = ibd.readline()

    # geneticMap = genetic_map(mapFile)
    # bps = geneticMap.breakpoints(STEP)
    # coverage = {}
    # coverage[chr] = [0]*(len(bps[chr])) #remember the last bin may not have full STEP length
    # #determine total length of IBD overalpping with each of the 0.25cM bins
    # for ibd in IBDset:
    #     start_i = bisect.bisect_left(bps[ibd.chr], ibd.start_bp)
    #     end_i = bisect.bisect_left(bps[ibd.chr], ibd.end_bp)
    #     if bps[ibd.chr][start_i] != ibd.start_bp:
    #         coverage[ibd.chr][start_i] += geneticMap.calcLengthCM(ibd.chr, \
    #             ibd.start_bp, bps[chr][start_i])
    #         #start_i += 1
    #     if end_i < len(bps[ibd.chr]) and bps[ibd.chr][end_i] != ibd.end_bp:
    #         coverage[ibd.chr][end_i-1] += geneticMap.calcLengthCM(ibd.chr, \
    #             bps[chr][end_i-1], ibd.end_bp)
    #         end_i -= 1
    #     elif end_i == len(bps[ibd_chr]):
    #         coverage[ibd.chr][-1] += tail[chr]
    #         end_i -= 1
        
    #     while start_i < end_i:
    #         coverage[ibd.chr][start_i] += STEP
    #         start_i += 1
    
    # #now normalize coverage and determine which regions are problematic
    # average = 0
    # for chr in bps.keys():
    #     average += sum(coverage[chr])
    #     coverage[chr][:-1] /= STEP
    #     coverage[chr][-1] /= tail[chr]
    
    # total_genome_length = sum([max(endMarker_cM[chr])-min(endMarker_cM[chr]) for chr in endMarker_cM.keys()])
    # average /= total_genome_length

    # print(f"average level of IBD sharing per centimorgan is {average}")
    # #print out problematic region
    # for chr in bps.keys():
    #     for i, cov in enumerate(coverage[chr]):
    #         if cov >= 5*average:
    #             end = max(endMarker_bp[chr]) if i == len(coverage[chr]) else bps[chr][i+1]
    #             print(f'{chr}:{bps[chr][i]}-{end} has unusually high IBD level. Excluded from further analysis.')



    ibdLen1 = [] #store length of IBDs that reside in the middle of a chromosome
    ibdLen2 = [] #store length of IBDs that reaches either end of a chromosome
    ibdseg_map1 = defaultdict(lambda: defaultdict(lambda: []))
    ibdseg_map2 = defaultdict(lambda: defaultdict(lambda: []))
    inds = set()
    with gzip.open(ibd_gz, 'rt') as ibd:
        line = ibd.readline()
        while line:
            ind1, hap1, ind2, hap2, chr, start_bp, end_bp, len_cM = line.strip().split('\t')
            chr, start_bp, end_bp, len_cM = int(chr), int(start_bp), int(end_bp), float(len_cM)
            ind1, ind2 = min(ind1, ind2), max(ind1, ind2)
            inds.add(ind1)
            inds.add(ind2)
            if start_bp in endMarker_bp[chr] or end_bp in endMarker_bp[chr]:
                ibdLen2.append(len_cM)
                ibdseg_map2[ind1][ind2].append(len_cM)
            else:
                ibdLen1.append(len_cM)
                ibdseg_map1[ind1][ind2].append(len_cM)
            line = ibd.readline()

    #calculate bins
    bin1, bin_midPoint1 = binning(ibdLen1)
    bin2, bin_midPoint2 = binning(ibdLen2)

    #return a vector of lengths (in cM) of each chromosome
    chr_len_cM = []
    for chr, start_end in endMarker_cM.items():
        chr_len_cM.append(abs(start_end[1]-start_end[0]))

    return bin1, bin2, bin_midPoint1, bin_midPoint2, np.array(chr_len_cM), inds, ibdseg_map1, ibdseg_map2


# def processIBDandBinning(ibd_gz, end_marker):
#     endMarker_bp = {}
#     endMarker_cM = {}

#     with open(end_marker) as end:
#         line = end.readline()
#         while line:
#             chr, rate, cM, phy = line.strip().split('\t')
#             chr, cM, phy = int(chr), float(cM), int(phy)
#             if not chr in endMarker_bp:
#                 endMarker_bp[chr] = [phy]
#                 endMarker_cM[chr] = [cM]
#             else:
#                 endMarker_bp[chr].append(phy)
#                 endMarker_cM[chr].append(cM)
#             line = end.readline()

#     total_genome_length = 0
#     for chr, start_end in endMarker_cM.items():
#         total_genome_length += abs(start_end[1]-start_end[0])

#     interval1 = np.arange(2,6,0.05)
#     interval2 = np.arange(6,8,0.1)
#     interval3 = np.arange(8,10,0.25)
#     interval4 = np.append(np.arange(10,100,5), 100)
#     bins = np.concatenate((interval1, interval2, interval3, interval4))

#     IBD_count_censored = np.zeros(len(bins))
#     IBD_count_uncensored = np.zeros(len(bins))

#     inds = set()
#     with gzip.open(ibd_gz, 'rt') as ibd:
#        line = ibd.readline()
#        while line:
#            ind1, hap1, ind2, hap2, chr, start_bp, end_bp, len_cM = line.strip().split('\t')
#            chr, start_bp, end_bp, len_cM = int(chr), int(start_bp), int(end_bp), float(len_cM)
#            inds.add(ind1)
#            inds.add(ind2)
#            tmp = bisect.bisect_left(bins, len_cM)
#            pos = tmp if (tmp < len(bins) and bins[tmp] == len_cM) else tmp-1
#            if start_bp in endMarker_bp[chr] or end_bp in endMarker_bp[chr]:
#                 IBD_count_censored[pos] += 1
#            else:
#                IBD_count_uncensored[pos] += 1
#            line = ibd.readline()

#     #now redistribute censored IBD into uncensored IBD
#     IBD_count_total = IBD_count_censored + IBD_count_uncensored
#     tmp = np.copy(IBD_count_uncensored)
#     frac = IBD_count_total/np.sum(IBD_count_total)
#     for b, count in enumerate(IBD_count_censored):
#         tmp[b:] += (count*frac[b:])/np.sum(frac[b:])

#     return tmp, bins, total_genome_length, len(inds)
