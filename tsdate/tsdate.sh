#!/usr/bin/env bash
#SBATCH --nodes=1
##SBATCH --ntasks=16
#SBATCH --ntasks=1
#SBATCH --mem=8000
#SBATCH --partition=regular
##SBATCH --nodelist=cbsubscb09
#SBATCH --job-name=tsdate
#SBATCH --output=tsdate.out.%j
#SBATCH --array=1-22
#SBATCH --mail-user=yh362@cornell.edu
#SBATCH --mail-type=ALL

host=$(hostname)
echo "On host $host"
date

chr=$SLURM_ARRAY_TASK_ID

if [ ! -d /fs/cbsubscb09/storage/yilei/simulate/chrom ]; then
  # need to mount cbsubscb09 storage
  /programs/bin/labutils/mount_server cbsubscb09 /storage
fi

python3 date.py -v ../../../simulate/chrom/EUR.chr$chr.maf.vcf -N 100000
