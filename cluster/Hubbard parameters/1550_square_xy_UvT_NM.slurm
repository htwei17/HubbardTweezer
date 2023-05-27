#!/bin/bash
#SBATCH --job-name="1550_3x3_square_xy_UvT_NM"
#SBATCH --partition=scavenge
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24GB
#SBATCH --mail-user=hw50@rice.edu
#SBATCH --mail-type=ALL
#

module getdefault
# conda initialization
__conda_setup="$('/opt/apps/software/Anaconda3/2021.11/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/apps/software/Anaconda3/2021.11/etc/profile.d/conda.sh" ]; then
        . "/opt/apps/software/Anaconda3/2021.11/etc/profile.d/conda.sh"
    else
        export PATH="/opt/apps/software/Anaconda3/2021.11/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate ~/env

N=20
R=3
Rz=7.2

FN=3D_3x3_square_xy_UvT_NM.ini
WORK_DIR=/scratch/hw50/HubbardTweezer/3D_3x3_square_xy_UvT_NM

mkdir -p $WORK_DIR
cp -r $SLURM_SUBMIT_DIR/src $WORK_DIR
cp $FN $WORK_DIR
echo "Job name: $SLURM_JOB_NAME"
echo "No. of cores to run: $SLURM_CPUS_PER_TASK"
echo "I ran on: $SLURM_NODELIST"
cd $WORK_DIR

if [ -s $FN ]; then
    echo "$FN is not empty. Nothing writen. Try to resume from interrupted result."
else
    echo "$FN is empty. Start writing parameters."
    echo "[Parameters]
N = $N
L0 = $R, $R, $Rz
lattice_size = 3, 3
lattice_const = 1550, 1600
V_0 = 52.26
waist = 1000,
laser_wavelength = 780
shape = square
scattering_length = 1770
dimension = 3
waist_direction = xy
lattice_symmetry = True
equalize = True
equalize_target = UvT
U_over_t = None
method = NM
write_log = True
no_bounds = False
verbosity = 3
job_id = $SLURM_JOB_ID" >>$FN
fi

# Code run
/home/hw50/env/bin/python -O -u src/Hubbard_exe.py $FN
cp $FN $SLURM_SUBMIT_DIR/output
