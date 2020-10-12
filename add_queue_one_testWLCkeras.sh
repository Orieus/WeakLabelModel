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

# Load Anaconda3 5.2.0
module load languages/python-anaconda3-5.2.0
# Load virtual environment
source ./venv/bin/activate

mkdir -p 'results'

datasets=gauss_quantiles

time python testWLCkeras.py --datasets ${datasets} --n-samples 2000 \
    --n-features 2 --n-classes 8 --n-simulations 10 --n-iterations 1000 \
    --loss square --mixing-matrix IPL --alpha 0.2 --beta 0.2
