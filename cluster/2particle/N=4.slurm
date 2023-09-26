#!/bin/bash
#SBATCH --partition=scavenge
#SBATCH --time=0:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1GB
#SBATCH --mail-user=hw50@rice.edu
#SBATCH --mail-type=ALL
#

module restore
# conda initialization
__conda_setup="$('/opt/apps/software/Anaconda3/2023.03-1/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/apps/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh" ]; then
        . "/opt/apps/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh"
    else
        export PATH="/opt/apps/software/Anaconda3/2023.03-1/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate $HOME/env

JOB_NAME="N=4_test"
fn=$JOB_NAME.ini
WORKING_LOCAL_PATH=$SHARED_SCRATCH/$USER/HubbardTweezer/$JOB_NAME

mkdir -p $WORKING_LOCAL_PATH
cp -r $SLURM_SUBMIT_DIR/src $WORKING_LOCAL_PATH
cp $SLURM_SUBMIT_DIR/cluster/2particle/$fn $WORKING_LOCAL_PATH
echo "I ran on: "$SLURM_NODELIST
# echo "Task No.: "$SLURM_ARRAY_TASK_ID

# Code run
cd $WORKING_LOCAL_PATH
$HOME/env/bin/python -O -u src/Hubbard_exe.py $fn
cp $fn $SLURM_SUBMIT_DIR