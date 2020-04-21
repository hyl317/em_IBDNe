#!/usr/bin/env bash
#SBATCH --nodes=1
##SBATCH --ntasks=16
#SBATCH --ntasks=1
#SBATCH --mem=6000
#SBATCH --partition=regular
##SBATCH --nodelist=cbsubscb09
##SBATCH --chdir=/fs/cbsubscb09/storage/yilei/simulate/chrom
#SBATCH --job-name=ibdne
#SBATCH --output=ibdne.eur.ts.out.%j
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

python3 em_IBDNe.py -i ../../../simulate/extractIBD/Dominic/EUR.ts.300.ibd.gz -e endMarker.ts.txt --alpha 0.01 -o eur.ts_300.300 --max_iter 500 -G 300 > test.eur.ts_300.300.tail
