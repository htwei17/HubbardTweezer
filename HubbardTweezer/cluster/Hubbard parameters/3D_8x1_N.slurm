#!/bin/bash
#SBATCH --job-name="3D_N"
#SBATCH --partition=scavenge
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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

fn=3D_8x1_N$SLURM_ARRAY_TASK_ID.ini
rm $fn

echo "[Parameters]" >>$fn
echo "N = "$SLURM_ARRAY_TASK_ID >>$fn
echo "L0 = 3, 3, 7.2
lattice = 8,
lattice_const = 1350, 1550
shape = square
scattering_length = 1600
laser_wavelength = 770
V_0 = 50
waist = 930, 1250
dimension = 3
waist_direction = xy
equalize = False
equalize_target = uvt
verbosity = 2" >>$fn

WORKING_LOCAL_PATH=$SHARED_SCRATCH/$USER/HubbardTweezer/"8x1_N="$SLURM_ARRAY_TASK_ID

mkdir -p $WORKING_LOCAL_PATH
cp -r $SLURM_SUBMIT_DIR/src $WORKING_LOCAL_PATH
cp $fn $WORKING_LOCAL_PATH
echo "I ran on: "$SLURM_NODELIST
echo "Task No.: "$SLURM_ARRAY_TASK_ID

# Code run
cd $WORKING_LOCAL_PATH
$HOME/env/bin/python -O -u src/Hubbard_exe.py $fn
