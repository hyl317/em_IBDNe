import tsinfer
import tsdate
import tskit
import argparse
import cyvcf2

def add_diploid_sites(vcf, samples):
    """
    Read the sites in the vcf and add them to the samples object, reordering the
    alleles to put the ancestral allele first, if it is available.
    """
    pos = 0
    for variant in vcf:  # Loop over variants, each assumed at a unique site
        if pos == variant.POS:
            raise ValueError("Duplicate positions for variant at position", pos)
        else:
            pos = variant.POS
        if any([not phased for _, _, phased in variant.genotypes]):
            raise ValueError("Unphased genotypes for variant at position", pos)
        alleles = [variant.REF] + variant.ALT
        ancestral = variant.INFO.get('AA', variant.REF)
        # Ancestral state must be first in the allele list.
        ordered_alleles = [ancestral] + list(set(alleles) - {ancestral})
        allele_index = {old_index: ordered_alleles.index(allele)
            for old_index, allele in enumerate(alleles)}
        # Map original allele indexes to their indexes in the new alleles list.
        genotypes = [allele_index[old_index]
            for row in variant.genotypes for old_index in row[0:2]]
        samples.add_site(pos, genotypes=genotypes, alleles=alleles)

def chromosome_length(vcf):
    assert len(vcf.seqlens) == 1
    return vcf.seqlens[0]

def main():
    parser = argparse.ArgumentParser(description='Use tsdate to infer node age')
    parser.add_argument('-v', action="store", dest="vcf", type=str, required=True, 
                        help='Path to VCF File')
    args = parser.parse_args()

    vcf = cyvcf2.VCF(args.vcf)
    with tsinfer.SampleData(path=f"{args.vcf}.samples") as samples:
        add_diploid_sites(vcf, samples)

    print("Sample file created for {} samples ".format(samples.num_samples) +
        "({} individuals) ".format(samples.num_individuals) +
        "with {} variable sites.".format(samples.num_sites), flush=True)

    # Do the inference
    ts = tsinfer.infer(samples)
    print("Inferred tree sequence: {} trees over {} Mb ({} edges)".format(
        ts.num_trees, ts.sequence_length/1e6, ts.num_edges))

if __name__ == '__main__':
    main()

