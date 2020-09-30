#!/bin/bash
#PBS -N logtestWLCkeras

#!/bin/bash
#PBS -N log_testWLCkeras
# request resources:
#PBS -l nodes=1:ppn=10
#PBS -l walltime=06:00:00

# on compute node, change directory to 'submission directory':
cd $PBS_O_WORKDIR
# record some potentially useful details about the job:
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo "PBS job ID is ${PBS_JOBID}"
echo "This jobs runs on the following machines:"
echo `cat $PBS_NODEFILE | uniq`
# count the number of processors available:
numprocs=`wc $PBS_NODEFILE | awk '{print $1}'`

mkdir -p 'results'

datasets=gauss_quantiles

time python testWLCkeras.py -p ${datasets} -s 2000 -f 2 -c 8 -m 10 \
        -i 1000 -l square -e IPL -t Mproper
