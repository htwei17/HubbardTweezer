#!/bin/bash
#SBATCH --job-name="3d_N"
#SBATCH --partition=scavenge
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32GB
#SBATCH --mail-user=hw50@rice.edu
#SBATCH --mail-type=ALL
#

module getdefault
# conda initialization
__conda_setup="$('/opt/apps/software/Anaconda3/2020.11/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
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
fi=100
ff=240
fs=10

# NORMAL WAIST
V0=104.52
w=1e-6
# FATTEST WAIST
# V0=156
# w=1.18e-6
# TIGHTEST WAIST
# V0=76
# w=8.61e-7

WORKING_LOCAL_PATH=$SHARED_SCRATCH/$USER/DVR/"N="$SLURM_ARRAY_TASK_ID # "_f="$fi","$ff","$fs

# Get different R's for fixed dx
r=$(echo "scale=20; $SLURM_ARRAY_TASK_ID*3/18" | bc)
rz=$(echo "scale=20; $r*2.4" | bc)

mkdir -p $WORKING_LOCAL_PATH
cp -r $SLURM_SUBMIT_DIR/*.py $WORKING_LOCAL_PATH
echo "I ran on: "$SLURM_NODELIST
echo "Task No.: "$SLURM_ARRAY_TASK_ID

# Code run
cd $WORKING_LOCAL_PATH
python3 -OO -u dynamics_exe.py -n $SLURM_ARRAY_TASK_ID -R $r $r $rz -d 3 -m Gaussian -f $fi $ff $fs -a -p $V0 $w
# python3 -OO -u dynamics_exe.py -n $SLURM_ARRAY_TASK_ID -d 3 -m Gaussian -f $fi $ff $fs -a
