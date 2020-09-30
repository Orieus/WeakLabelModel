#!/bin/bash
#PBS -N logtestWLCkeras

#!/bin/bash
#PBS -N log_testWLCkeras
# request resources:
#PBS -l nodes=1:ppn=10
#PBS -l walltime=6:00:00

## Options to run job arrays
#PBS -t 0-32

# on compute node, change directory to 'submission directory':
cd $PBS_O_WORKDIR
# record some potentially useful details about the job:
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo "PBS job ID is ${PBS_JOBID}"
echo "The Array ID is: ${PBS_ARRAYID}"
echo "This jobs runs on the following machines:"
echo `cat $PBS_NODEFILE | uniq`
# count the number of processors available:
numprocs=`wc $PBS_NODEFILE | awk '{print $1}'`

mkdir -p 'results'

declare -a dataset_names=(
    'confidence'
    'segment'
    'satimage'
    'JapaneseVowels'
    'pendigits'
    'iris'
    'blobs'
    'fl2000'
    'yeast'
    'page-blocks'
    'gauss_quantiles'
    'autoUniv-au6-750'
    'abalone'
    'GesturePhaseSegmentationProcessed'
    'autoUniv-au6-1000'
    'vowel'
    'visualizing_livestock'
    'vehicle'
    'analcatdata_dmft'
    'cardiotocography'
    'car'
    'diggle_table_a2'
    'ecoli'
    'flags'
    'glass'
    'balance-scale'
    'wine'
    'autoUniv-au7-1100'
    'autoUniv-au7-500'
    'mfeat-zernike'
    'collins'
    'prnn_fglass'
    'zoo'
    )

datasets=${dataset_names[$PBS_ARRAYID]}

time python testWLCkeras.py -p ${datasets} -s 1000 -f 2 -c 5 -m 10 \
        -i 40 -l square -e random_weak \
        -t Mproper -a 1.0 -b 0.0 -u results_a10_b0
