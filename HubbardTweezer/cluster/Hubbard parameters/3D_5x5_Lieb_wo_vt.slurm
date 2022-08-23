#!/bin/bash
#SBATCH --job-name="3D_5x5_Lieb_wo_vt"
#SBATCH --partition=scavenge
#SBATCH --time=4:00:00
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

fn=3D_5x5_Lieb_wo_vt.ini
rm $fn

echo "[Parameters]" >>$fn
echo "N = 20
L0 = 3, 3, 7.2
lattice = 5, 5
lattice_const = 1520, 1690
shape = Lieb
scattering_length = 1000
laser_wavelength = 780
V_0 = 104.52
average = 0.5
waist = 1000, 1000
dimension = 3
waist_direction = None
equalize = True
equalize_target = vt
verbosity = 2" >>$fn

WORKING_LOCAL_PATH=$SHARED_SCRATCH/$USER/HubbardTweezer/"5x5_Lieb_wo_vt"

mkdir -p $WORKING_LOCAL_PATH
cp -r $SLURM_SUBMIT_DIR/src $WORKING_LOCAL_PATH
cp $fn $WORKING_LOCAL_PATH
echo "I ran on: "$SLURM_NODELIST
# echo "Task No.: "$SLURM_ARRAY_TASK_ID

# Code run
cd $WORKING_LOCAL_PATH
$HOME/env/bin/python -O -u src/Hubbard_exe.py $fn
cp $fn $SLURM_SUBMIT_DIR