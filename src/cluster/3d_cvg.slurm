#!/bin/bash
#SBATCH --job-name="3d_tweezer_cvg"
#SBATCH --partition=commons
#SBATCH --time=24:00:00
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
fi=0.02
ff=0.30
fs=0.02
WORKING_LOCAL_PATH=$SHARED_SCRATCH/$USER/DVR/"N="$SLURM_ARRAY_TASK_ID"_f="$fi","$ff","$fs
mkdir -p $WORKING_LOCAL_PATH
cp -r $SLURM_SUBMIT_DIR/*.py $WORKING_LOCAL_PATH
echo "I ran on: "$SLURM_NODELIST
echo "Task No.: "$SLURM_ARRAY_TASK_ID

# Code run
cd $WORKING_LOCAL_PATH
python3 -u dynamics_exe.py -n $SLURM_ARRAY_TASK_ID -d 3 -m Gaussian -f $fi $ff $fs

# Final steps
# rm -r $WORKING_LOCAL_PATH/*.out
cd $SLURM_SUBMIT_DIR
mkdir -p ../output
cp -r $WORKING_LOCAL_PATH/*.h5 ../output
# rm -r $WORKING_LOCAL_PATH/*