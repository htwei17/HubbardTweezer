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
RAND=False
RAND_SAMPLE=False
WAIST=xy
STATUS=neq
PARTITION=scavenge
TIME="04:00:00"
ARRAY="1-1"
ARRAY_SUFFIX=""
LOG=True
METHOD="trf"
# DVR
d=3
N=20
R=3
Rz=7.2
SYMMETRY=True
GHOST=False
GHOST_PARAM=""
SUB_PATH=""
NL_DEFINITION="N=$N
R=$R
Rz=$Rz"

# ====== Read arguments ======
readArg() {
    if [ "$2" ]; then
        local param=$2
        echo $param
    else
        die 'ERROR: "$1" requires a non-empty option argument.'
    fi
}

while :; do
    case $1 in
    -h | -\? | --help)
        # Display a usage synopsis.
        echo "HELP: Hubbard parameter job submission"
        echo "-l, --L:              lattice grid x size (Default: $Lx)"
        echo "-ly, --Ly:            lattice grid y size (Default: $Ly)"
        echo "-ld, --lattice-dim:   lattice dimension (Default: $LATTICE_DIM)"
        echo "-d, --D:              DVR dimension (Default: $d)"
        echo "-s, --shape:          lattice shape (Default: $SHAPE)"
        echo "-w, --waist:          determine which waist direction to vary (Default: $WAIST)"
        echo "                      it can be 'x', 'y', 'xy' and 'None'"
        echo "-e, --eq:             determine which parameter to equalize (Default: $STATUS)"
        echo "                      it can be 'neq' for no equalization,"
        echo "                      'L'('N') for varying L(N) to check convergence ('neq' implied)"
        echo "-r, --random:         to use random initial guess or not (Default: $RAND)"
        echo "-u, --Ut:             Hubbard parameter U/t (Default: $Ut)"
        echo "-sy, --symmetry:      to use lattice symmetry or not (DefaultL $SYMMETRY)"
        echo "-g, --ghost:          to use ghost traps or not (Default: $GHOST)"
        echo "-m, --method:         method used to minimize cost function (Default: $METHOD)"
        echo "                      it can be 'trf', 'Nelder-Mead', 'bfgs', 'slsqp', 'bobyqa', 'direct', 'crs2', 'subplex'"
        echo "-a,--array:           array job ID range (Default: $ARRAY)"
        exit
        ;;
    -l | --L) # Takes an option argument; ensure it has been specified.
        Lx=$(readArg --L $2)
        Ly=$Lx
        shift
        ;;
    -ly | --Ly) # Takes an option argument; ensure it has been specified.
        Ly=$(readArg --Ly $2)
        shift
        ;;
    -ld | --lattice-dim)
        LATTICE_DIM=$(readArg --lattice-dim $2)
        shift
        ;;
    -d | --D)
        d=$(readArg --D $2)
        shift
        ;;
    -s | --shape)
        SHAPE=$(readArg --shape $2)
        shift
        ;;
    -w | --waist)
        WAIST=$(readArg --waist $2)
        shift
        ;;
    -e | --eq)
        STATUS=$(readArg --eq $2)
        shift
        ;;
    -r | --random)
        RAND=$(readArg --random $2)
        shift
        ;;
    -u | --Ut)
        Ut=$(readArg --Ut $2)
        shift
        ;;
    -sy | --symmetry)
        SYMMETRY=$(readArg --symmetry $2)
        shift
        ;;
    -g | --ghost)
        GHOST=$(readArg --ghost $2)
        shift
        ;;
    -m | --method)
        METHOD=$(readArg --method $2)
        shift
        ;;
    -a | --array)
        ARRAY=$(readArg --array $2)
        shift
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

# ========= Methods =========
METHOD_SUFFIX="_"$METHOD

# If job is not finished w/i scavenge wall time, just continue by using the same ini
if [ $METHOD = "NM" ] || [ $METHOD = "Nelder-Mead" ] || [ $METHOD = "subplex" ] || [ $METHOD = "direct" ] || [ $METHOD = "crs2" ] || [ $METHOD = "bobyqa" ] || [ $METHOD = "praxis" ] || [ $WAIST != "None" ]; then
    EQ_FLAG=True
    PARTITION=commons
    TIME="24:00:00"
fi

# ========= Lattice =========
if [ $SHAPE = "square" ] && [ $Ly = 1 ]; then
    LATTICE_DIM=1
elif [ $SHAPE != "square" ]; then
    LATTICE_DIM=2
fi

if [ $LATTICE_DIM -ge 2 ] && [ $SHAPE = 'triangular' ]; then
    # 2D triangular
    DIM_PARAM="lattice_size = $Lx, $Ly
lattice_const = 1550,
V0 = 73.0219"
elif [ $LATTICE_DIM -ge 2 ] && [ $SHAPE = 'zigzag' ]; then
    # 2D triangular
    DIM_PARAM="lattice_size = $Lx, $Ly
lattice_const = 2400, 1000
V0 = 52.26"
    SYMMETRY=False
elif [ $LATTICE_DIM -ge 2 ] && [ $SHAPE != 'ring' ]; then
    # 2D other lattice
    DIM_PARAM="lattice_size = $Lx, $Ly
lattice_const = 1550, 1600
V0 = 52.26"
elif [ $SHAPE = 'ring' ]; then
    # Ring
    # Build a perfect ring s.t. no equalization needed
    Ly=1
    DIM_PARAM="lattice_size = $Lx, $Ly
lattice_const = 1550,
V0 = 52.26"
else
    # 1D chain
    Ly=1
    TIME="00:40:00"
    DIM_PARAM="lattice_size = $Lx,
lattice_const = 1550,
V0 = 52.26"
fi

# ========= Non-equalization =========
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
    TIME="0:02:00"
    METHOD_SUFFIX=""
    ARRAY="12-22:2"
    ARRAY_SUFFIX="_\$SLURM_ARRAY_TASK_ID"
    NL_DEFINITION="N=\$SLURM_ARRAY_TASK_ID
R=\$(echo \"scale=20; \$SLURM_ARRAY_TASK_ID*3/20\" | bc)
Rz=\$(echo \"scale=20;\$R*2.4\" | bc)"
elif [ $STATUS = "N" ]; then
    EQ_FLAG=False
    WAIST=None
    PARTITION=scavenge
    TIME="0:02:00"
    METHOD_SUFFIX=""
    ARRAY="12-22:2"
    ARRAY_SUFFIX="_\$SLURM_ARRAY_TASK_ID"
    NL_DEFINITION="N=\$SLURM_ARRAY_TASK_ID
R=$R
Rz=$Rz"
fi

# ========= Waist trap =========
if [ $GHOST = True ]; then
    SUB_PATH="/ghost"
    WAIST=None
    GHOST_PARAM="
ghost_penalty = 1, 1"
fi

if [[ $EQ_FLAG == "False" ]] && [[ $RAND == "True" ]]; then
    RAND_SAMPLE=True
    ARRAY_SUFFIX="_\$SLURM_ARRAY_TASK_ID"
    SUB_PATH="/samples"
fi

# ========= Write sbatch script =========
echo "Lattice size is: $Lx,$Ly"
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
__conda_setup=\"\$('/opt/apps/software/Anaconda3/2022.10/bin/conda' 'shell.bash' 'hook' 2>/dev/null)\"
if [ \$? -eq 0 ]; then
    eval \"\$__conda_setup\"
else
    if [ -f \"/opt/apps/software/Anaconda3/2022.10/etc/profile.d/conda.sh\" ]; then
        . \"/opt/apps/software/Anaconda3/2022.10/etc/profile.d/conda.sh\"
    else
        export PATH=\"/opt/apps/software/Anaconda3/2022.10/bin:\$PATH\"
    fi
fi
unset __conda_setup
conda activate ~/env

$NL_DEFINITION

FN=$JOB_NAME$ARRAY_SUFFIX.ini
WORK_DIR=$SHARED_SCRATCH/$USER/HubbardTweezer$SUB_PATH/$JOB_NAME$ARRAY_SUFFIX

mkdir -p \$WORK_DIR
cp -r \$SLURM_SUBMIT_DIR/src \$WORK_DIR
cd \$WORK_DIR

if [ -s \$FN ]; then
    echo \"\$FN is not empty. Nothing writen. Try to resume from interrupted result.\"
else
    echo \"\$FN is empty. Start writing parameters.\"
    echo \"[Parameters]
N = \$N
L0 = \$R, \$R, \$Rz
$DIM_PARAM
waist = 1000,
laser_wavelength = 780
shape = $SHAPE
scattering_length = 1770
dimension = $d
lattice_symmetry = $SYMMETRY
equalize = $EQ_FLAG
equalize_target = $STATUS
U_over_t = $Ut
method = $METHOD
random_initial_guess = $RAND
ghost_sites = $GHOST$GHOST_PARAM
waist_direction = $WAIST
write_log = $LOG
no_bounds = False
verbosity = 3
job_id = \$SLURM_JOB_ID\" >>\$FN
fi

echo \"Job name: \$SLURM_JOB_NAME\"
echo \"No. of cores to run: \$SLURM_CPUS_PER_TASK\"
echo \"I ran on: \$SLURM_NODELIST\"
# echo \"Task No.: \$SLURM_ARRAY_TASK_ID\"

# Code run
$HOME/env/bin/python -O -u src/Hubbard_exe.py \$FN
cp \$FN \$SLURM_SUBMIT_DIR/output$SUB_PATH" >>$SLURM_FN

# ========= Run sbatch =========
if [[ $STATUS == "L" ]] || [[ $STATUS == "N" ]] || [[ $RAND_SAMPLE == "True" ]]; then
    sbatch --array=$ARRAY $SLURM_FN
else
    sbatch $SLURM_FN
fi
