#!/bin/bash

fi=0.12
ff=0.32
fs=0.02
FREQ="f="$fi","$ff","$fs
GROUP_WORK_PATH=$WORK/kh30/hw50

# sbatch --array=12-16:2 3d_cvg_s.slurm
# sbatch --array=18 3d_cvg_l.slurm

cd $GROUP_WORK_PATH
mkdir -p DVR/output
for ((i = 12; i <= 20; i += 2)); do
    WORKING_LOCAL_PATH=$SHARED_SCRATCH/$USER/DVR/"N="$i"_"$FREQ
    cp $WORKING_LOCAL_PATH/*.h5 DVR/output
done

# N=18
# for ((i = 2; i <= 30; i += 2)); do
#     freq=0.$(printf "%02d" $i)
#     WORKING_LOCAL_PATH=$SHARED_SCRATCH/$USER/DVR/"N="$N"_f="$freq
#     cp -r $WORKING_LOCAL_PATH/*.h5 DVR/output
# done
