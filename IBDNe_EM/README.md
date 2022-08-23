# em_IBDNe

An IBDNe-like approach to estimate recnet population size history from IBD segments.

## Usage
To view all available command line options, use

    python EM_moment.py -h

Required arguments are -i, -e, -n.

-i is the [HapIBD](https://github.com/browning-lab/hap-ibd) output file.
-e is a txt file that describes the first and last SNP marker of each chromosome. This should match the .vcf file used for detecting IBD segments. An example of this can be found at ./samples/endMarker.averageMap.txt
-n gives the number of samples. This is important as occasioanally a sample may not share any IBD with anyone else, so sample size cannot be deduced from the IBD file.