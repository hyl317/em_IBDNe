#!/usr/bin/env bash
#SBATCH --nodes=1
##SBATCH --ntasks=16
#SBATCH --ntasks=1
#SBATCH --mem=6000
#SBATCH --partition=regular
##SBATCH --nodelist=cbsubscb09
##SBATCH --chdir=/fs/cbsubscb09/storage/yilei/simulate/chrom
#SBATCH --job-name=ibdne_likelihood
#SBATCH --output=ibdne.likelihood.BFGS.diff2.out.%j
##SBATCH --array=1-22
#SBATCH --mail-user=yh362@cornell.edu
#SBATCH --mail-type=ALL

host=$(hostname)
echo "On host $host"
date

if [ ! -d /fs/cbsubscb09/storage/yilei/simulate/chrom ]; then
  # need to mount cbsubscb09 storage
  /programs/bin/labutils/mount_server cbsubscb09 /storage
fi

#python3 effective_sample_size.py -e endMarker.ts.txt -n 2000 --ibd ../../../simulate/extractIBD/Dominic/EUR.ts.200.ibd.gz --hbd ../../../simulate/extractIBD/Dominic/EUR.ts.200.hbd.gz --bins 2,2.5,3,3.5,4,4.5,5,5.5,6,7,8,9,10,15,20,30,40,50 -N ~/EUR.ref.ne.200.txt

python3 effective_sample_size.py -e endMarker.ts.txt -n 2000 --ibd ../../../simulate/extractIBD/Dominic/EUR.ts.200.ibd.gz --hbd ../../../simulate/extractIBD/Dominic/EUR.ts.200.hbd.gz -N ~/EUR.ref.ne.200.txt
