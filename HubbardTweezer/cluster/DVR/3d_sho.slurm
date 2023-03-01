#!/bin/bash
#SBATCH --job-name="3d_sho"
#SBATCH --partition=scavenge
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128GB
#SBATCH --mail-user=hw50@rice.edu
#SBATCH --mail-type=ALL
#

module getdefault
# conda initialization
__conda_setup="$('/opt/apps/software/Anaconda3/2020.11/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/apps/software/Anaconda3/2020.11/etc/profile.d/conda.sh" ]; then
        . "/opt/apps/software/Anaconda3/2020.11/etc/profile.d/conda.sh"
    else
        export PATH="/opt/apps/software/Anaconda3/2020.11/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate ~/env

# File preparations
WORKING_LOCAL_PATH=$SHARED_SCRATCH/$USER/DVR
mkdir -p $WORKING_LOCAL_PATH
cp -r $SLURM_SUBMIT_DIR/*.py $WORKING_LOCAL_PATH
echo "I ran on: "$SLURM_NODELIST
echo "Task No.: "$SLURM_ARRAY_TASK_ID

# Code run
cd $WORKING_LOCAL_PATH
# ulimit
python3 -u dynamics_exe.py -n 15 -d 3 -m sho -f 0.$(printf "%02d" $SLURM_ARRAY_TASK_ID) 0.$(printf "%02d" $SLURM_ARRAY_TASK_ID) 0.01

# Final steps
# rm -r $WORKING_LOCAL_PATH/*.out
# cd $SLURM_SUBMIT_DIR
# mkdir -p ../output
# cp -r $WORKING_LOCAL_PATH/*.csv ../output
GROUP_WORK_PATH=/storage/hpc/work/kh30/hw50
cd $GROUP_WORK_PATH
mkdir -p DVR/output
cp -r $WORKING_LOCAL_PATH/*.h5 DVR/output
# rm -r $WORKING_LOCAL_PATH/*