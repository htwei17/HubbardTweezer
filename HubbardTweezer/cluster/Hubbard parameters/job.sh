#!/bin/bash

# ====== Default values ======
# Lattice
L=3
LATTICE_DIM=2
SHAPE=square
# Equalization
EQ_FLAG=True
WAIST=xy
STATUS=neq
PARTITION=scavenge
TIME="4:00:00"
SUFFIX=""
# DVR
d=3
N=20
R=3
Rz=7.2
NL_DEFINITION="N=$N
R=$R
Rz=$Rz"

# ====== Read arguments ======
while :; do
    case $1 in
    -h|-\?|--help)
    # Display a usage synopsis.
        echo "HELP: Hubbard parameter job submission"
        echo "-l, --L:  lattice grid size (Default: $L)"
        echo "-t, --lattice-dim:    lattice dimension (Default: $LATTICE_DIM)"
        echo "-d, --D:  DVR dimension (Default: $d)"
        echo "-s, --shape:  lattice shape (Default: $s)"
        echo "-w, --waist:  determine which waist direction to vary (Default: $WAIST)"
        echo "              it can be 'x', 'y', 'xy' and 'None'"
        echo "-e, --eq: determine which parameter to equalize (Default: $STATUS)"
        echo "          it can be 'neq' for no equalization,"
        echo "          'L'('N') for varying L(N) to check convergence ('neq' implied)"
        exit
        ;;
    -l | --L) # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            L=$2
            shift
        else
            die 'ERROR: "--L" requires a non-empty option argument.'
        fi
        ;;
    -t | --lattice-dim)
        if [ "$2" ]; then
            LATTICE_DIM=$2
            shift
        else
            die 'ERROR: "--d" requires a non-empty option argument.'
        fi
        ;;
    -d | --D)
        if [ "$2" ]; then
            d=$2
            shift
        else
            die 'ERROR: "--d" requires a non-empty option argument.'
        fi
        ;;
    -s | --shape)
        if [ "$2" ]; then
            SHAPE=$2
            shift
        else
            die 'ERROR: "--shape" requires a non-empty option argument.'
        fi
        ;;
    -w | --waist)
        if [ "$2" ]; then
            WAIST=$2
            shift
        else
            die 'ERROR: "--waist" requires a non-empty option argument.'
        fi
        ;;
    -e | --eq)
        if [ "$2" ]; then
            STATUS=$2
            shift
        else
            die 'ERROR: "--eq" requires a non-empty option argument.'
        fi
        ;;
    --) # End of all options.
        shift
        break
        ;;
    -?*)
        printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
        ;;
    *) # Default case: No more options, so break out of the loop.
        break ;;
    esac
    shift
done

if [[ $STATUS == "neq" ]]; then
    EQ_FLAG=False
    WAIST=None
    PARTITION=scavenge
    TIME="0:02:00"
elif [[ $STATUS == "L" ]]; then
    EQ_FLAG=False
    WAIST=None
    PARTITION=scavenge
    TIME="0:02:00"
    SUFFIX="_\$SLURM_ARRAY_TASK_ID"
    NL_DEFINITION="N=$N
R=\$(echo \"scale=20; \$SLURM_ARRAY_TASK_ID*3/20\" | bc)
Rz=\$(echo \"scale=20;\$R*2.4\" | bc)"
elif [[ $STATUS == "N" ]]; then
    EQ_FLAG=False
    WAIST=None
    PARTITION=scavenge
    TIME="0:02:00"
    SUFFIX="_\$SLURM_ARRAY_TASK_ID"
    NL_DEFINITION="N=\$SLURM_ARRAY_TASK_ID
R=$R
Rz=$Rz"
fi

# if [ $WAIST != "None" ]; then
#     EQ_FLAG=True
#     PARTITION=commons
#     TIME="12:00:00"
# fi

if [ $SHAPE != "square" ]; then
    LATTICE_DIM=2
fi

if [ $LATTICE_DIM -ge 2 ]; then
    Lx=$L
    Ly=$L
    DIM_PARAM="lattice = $Lx, $Ly
lattice_const = 1520, 1690
laser_wavelength = 780
V_0 = 52.26
waist = 1000, 1000"
else
    Lx=$L
    Ly=1
    DIM_PARAM="lattice = $Lx,
lattice_const = 1350, 1550
laser_wavelength = 770
V_0 = 50
waist = 930, 1250"
fi

JOB_NAME=$d"D_"$Lx"x"$Ly"_"$SHAPE"_"$WAIST"_"$STATUS

SLURM_FN=$JOB_NAME.slurm
rm $SLURM_FN

echo "#!/bin/bash
#SBATCH --job-name=\"$JOB_NAME\"
#SBATCH --partition=$PARTITION
#SBATCH --time=$TIME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32GB
#SBATCH --mail-user=hw50@rice.edu
#SBATCH --mail-type=ALL
#

module getdefault
# conda initialization
__conda_setup=\"\$('/opt/apps/software/Anaconda3/2021.11/bin/conda' 'shell.bash' 'hook' 2>/dev/null)\"
if [ \$? -eq 0 ]; then
    eval \"\$__conda_setup\"
else
    if [ -f \"/opt/apps/software/Anaconda3/2021.11/etc/profile.d/conda.sh\" ]; then
        . \"/opt/apps/software/Anaconda3/2021.11/etc/profile.d/conda.sh\"
    else
        export PATH=\"/opt/apps/software/Anaconda3/2021.11/bin:\$PATH\"
    fi
fi
unset __conda_setup
conda activate ~/env

$NL_DEFINITION

FN=$JOB_NAME$SUFFIX.ini
if [ -s \$FN ]; then
    echo \"\$FN is not empty. No parameter section writen.\"
else
    echo \"\$FN is empty. Start writing parameters.\"
    echo \"[Parameters]
N = \$N
L0 = \$R, \$R, \$Rz
$DIM_PARAM
shape = $SHAPE
scattering_length = 1000
dimension = $d
waist_direction = $WAIST
equalize = $EQ_FLAG
equalize_target = $STATUS
verbosity = 2\" >>\$FN
fi

WORK_DIR=$SHARED_SCRATCH/$USER/HubbardTweezer/$JOB_NAME$SUFFIX

mkdir -p \$WORK_DIR
cp -r \$SLURM_SUBMIT_DIR/src \$WORK_DIR
cp \$FN \$WORK_DIR
echo \"Job name: \$SLURM_JOB_NAME\"
echo \"No. of cores to run: \$SLURM_CPUS_PER_TASK\"
echo \"I ran on: \$SLURM_NODELIST\"
# echo \"Task No.: \$SLURM_ARRAY_TASK_ID\"

# Code run
cd \$WORK_DIR
$HOME/env/bin/python -O -u src/Hubbard_exe.py \$FN
cp \$FN \$SLURM_SUBMIT_DIR" >>$SLURM_FN

if [[ $STATUS == "L" ]] || [[ $STATUS == "N" ]]; then
    sbatch --array=16-22:2 $SLURM_FN
else
    sbatch --export=L=$L $SLURM_FN
fi
