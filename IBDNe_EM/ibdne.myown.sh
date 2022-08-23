#!/usr/bin/env bash
#SBATCH --nodes=1
##SBATCH --ntasks=16
#SBATCH --ntasks=1
#SBATCH --mem=6000
#SBATCH --partition=regular
##SBATCH --nodelist=cbsubscb09
##SBATCH --chdir=/fs/cbsubscb09/storage/yilei/simulate/chrom
#SBATCH --job-name=ibdne
#SBATCH --output=ibdne.fin.MS.%j
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

python em_IBDNe.py -i /fs/cbsubscb09/storage/yilei/simulate/MS_ALL/MS.fin.vcf/FIN.MS.unr.str.ibd.gz -e /fs/cbsubscb09/storage/yilei/simulate/MS_ALL/MS.fin.vcf/endMarker.MS.fin --alpha 0.001 -o fin.MS.200.str --max_iter 250 -G 200 -n 535 --minIBD 2.5
