#!/bin/bash

# ====== Default values ======
# Lattice
Lx=3
Ly=$Lx
LATTICE_DIM=2
SHAPE=square
# Equalization
Ut=None
EQ_FLAG=True
WAIST=xy
STATUS=neq
PARTITION=scavenge
TIME="4:00:00"
LN_SUFFIX=""
LOG=True
METHOD="trf"
# DVR
d=3
N=20
R=3
Rz=7.2
SYMMETRY=True
NL_DEFINITION="N=$N
R=$R
Rz=$Rz"

# ====== Read arguments ======
while :; do
    case $1 in
    -h | -\? | --help)
        # Display a usage synopsis.
        echo "HELP: Hubbard parameter job submission"
        echo "-l, --L:  lattice grid x size (Default: $Lx)"
        echo "-y, --Ly:  lattice grid y size (Default: $Ly)"
        echo "-t, --lattice-dim:    lattice dimension (Default: $LATTICE_DIM)"
        echo "-d, --D:  DVR dimension (Default: $d)"
        echo "-s, --shape:  lattice shape (Default: $s)"
        echo "-w, --waist:  determine which waist direction to vary (Default: $WAIST)"
        echo "              it can be 'x', 'y', 'xy' and 'None'"
        echo "-e, --eq: determine which parameter to equalize (Default: $STATUS)"
        echo "          it can be 'neq' for no equalization,"
        echo "          'L'('N') for varying L(N) to check convergence ('neq' implied)"
        echo "-u, --Ut:  Hubbard parameter U/t (Default: $Ut)"
        echo "-v, --symmetry: to use lattice symmetry or not (DefaultL $SYMMETRY)"
        echo "-m, --method: method used to minimize cost function (Default: $METHOD)"
        exit
        ;;
    -l | --L) # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            Lx=$2
            Ly=$Lx
            shift
        else
            die 'ERROR: "--L" requires a non-empty option argument.'
        fi
        ;;
    -y | --Ly) # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            Ly=$2
            shift
        else
            die 'ERROR: "--Ly" requires a non-empty option argument.'
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
    -u | --Ut)
        if [ "$2" ]; then
            Ut=$2
            shift
        else
            die 'ERROR: "--Ut" requires a non-empty option argument.'
        fi
        ;;
    -v | --symmetry)
        if [ "$2" ]; then
            SYMMETRY=$2
            shift
        else
            die 'ERROR: "--symmetry" requires a non-empty option argument.'
        fi
        ;;
    -m | --method)
        if [ "$2" ]; then
            METHOD=$2
            shift
        else
            die 'ERROR: "--method" requires a non-empty option argument.'
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

echo "Lattice size is: $Lx,$Ly"
METHOD_SUFFIX="_"$METHOD

if [ $STATUS = "neq" ]; then
    EQ_FLAG=False
    WAIST=None
    PARTITION=scavenge
    LOG=False
    METHOD_SUFFIX=""
    if [ $SYMMETRY = "True" ]; then
        TIME="0:05:00"
    else
        TIME="0:20:00"
    fi
elif [ $STATUS = "L" ]; then
    EQ_FLAG=False
    WAIST=None
    PARTITION=scavenge
    TIME="0:05:00"
    METHOD_SUFFIX=""
    LN_SUFFIX="_\$SLURM_ARRAY_TASK_ID"
    NL_DEFINITION="N=$N
R=\$(echo \"scale=20; \$SLURM_ARRAY_TASK_ID*3/20\" | bc)
Rz=\$(echo \"scale=20;\$R*2.4\" | bc)"
elif [ $STATUS = "N" ]; then
    EQ_FLAG=False
    WAIST=None
    PARTITION=scavenge
    TIME="0:05:00"
    LN_SUFFIX="_\$SLURM_ARRAY_TASK_ID"
    NL_DEFINITION="N=\$SLURM_ARRAY_TASK_ID
R=$R
Rz=$Rz"
fi

if [ $METHOD = "NM" ] || [ $METHOD = "Nelder-Mead" ]; then
    PARTITION=commons
    TIME="8:00:00"
fi

# if [ $WAIST != "None" ]; then
#     EQ_FLAG=True
#     PARTITION=commons
#     TIME="12:00:00"
# fi

if [ $SHAPE != "square" ]; then
    LATTICE_DIM=2
fi

if [ $LATTICE_DIM -ge 2 ] && [ $SHAPE = 'triangular' ]; then
    DIM_PARAM="lattice_size = $Lx, $Ly
lattice_const = 1550,
laser_wavelength = 780
V_0 = 73.0219
waist = 1000, 1000"
elif [ $LATTICE_DIM -ge 2 ] && [ $SHAPE != 'ring' ]; then
    DIM_PARAM="lattice_size = $Lx, $Ly
lattice_const = 1520, 1690
laser_wavelength = 780
V_0 = 52.26
waist = 1000, 1000"
elif [ $SHAPE = 'ring' ]; then
    # Build a perfect ring s.t. no equalization needed
    Ly=1
    DIM_PARAM="lattice_size = $Lx, $Ly
lattice_const = 1520, 1520
laser_wavelength = 780
V_0 = 52.26
waist = 1000, 1000"
else
    Lx=$L
    Ly=1
    DIM_PARAM="lattice_size = $Lx,
lattice_const = 1350, 1550
laser_wavelength = 770
V_0 = 50
waist = 930, 1250"
fi

JOB_NAME=$d"D_"$Lx"x"$Ly"_"$SHAPE"_"$WAIST"_"$STATUS$METHOD_SUFFIX

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

FN=$JOB_NAME$LN_SUFFIX.ini
if [ -s \$FN ]; then
    echo \"\$FN is not empty. Nothing writen. Try to resume from interrupted result.\"
else
    echo \"\$FN is empty. Start writing parameters.\"
    echo \"[Parameters]
N = \$N
L0 = \$R, \$R, \$Rz
$DIM_PARAM
shape = $SHAPE
scattering_length = 1770
dimension = $d
waist_direction = $WAIST
lattice_symmetry = $SYMMETRY
equalize = $EQ_FLAG
equalize_target = $STATUS
U_over_t = $Ut
method = $METHOD
write_log = $LOG
no_bounds = False
verbosity = 3
job_id = \$SLURM_JOB_ID\" >>\$FN
fi

WORK_DIR=$SHARED_SCRATCH/$USER/HubbardTweezer/$JOB_NAME$LN_SUFFIX

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
cp \$FN \$SLURM_SUBMIT_DIR/output" >>$SLURM_FN

if [[ $STATUS == "L" ]] || [[ $STATUS == "N" ]]; then
    sbatch --array=16-22:2 $SLURM_FN
else
    sbatch --export=L=$L $SLURM_FN
fi
