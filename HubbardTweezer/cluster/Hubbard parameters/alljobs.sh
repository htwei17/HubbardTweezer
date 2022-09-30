#!/bin/bash

rm *.slurm *.ini

bash job.sh -t 1 -l 8 -w None -e UT
bash job.sh -t 1 -l 8 -w None -e UvT
bash job.sh -t 1 -l 8 -w None -e Uv
bash job.sh -t 1 -l 8 -w None -e vT
bash job.sh -t 1 -l 8 -w x -e vT
bash job.sh -t 1 -l 8 -w x -e UvT
bash job.sh -t 1 -l 8 -w x -e Uv
bash job.sh -t 1 -l 8 -w x -e UT

bash job.sh -w None -e UT
bash job.sh -w None -e UvT
bash job.sh -w None -e Uv
bash job.sh -w None -e vT
bash job.sh -w xy -e vT
bash job.sh -w xy -e UvT
bash job.sh -w xy -e Uv
bash job.sh -w xy -e UT

bash job.sh -s Lieb -w None -e UT
bash job.sh -s Lieb -w None -e UvT
bash job.sh -s Lieb -w None -e Uv
bash job.sh -s Lieb -w None -e vT
bash job.sh -s Lieb -w xy -e vT
bash job.sh -s Lieb -w xy -e UvT
bash job.sh -s Lieb -w xy -e Uv
bash job.sh -s Lieb -w xy -e UT

# bash job.sh -s triangular -l 4 -y 2 -v False -w None -e UvT
# bash job.sh -s zigzag -l 4 -y 2 -v False -w None -e neq
# bash job.sh -s zigzag -l 4 -y 2 -v False -w None -e vT
# bash job.sh -s zigzag -l 4 -y 2 -v False -w None -e UvT
# bash job.sh -s zigzag -l 4 -y 2 -v False -w xy -e UvT

# bash job1600.sh -w None -e UT -m NM
# bash job1600.sh -w None -e UvT -m NM
# bash job1600.sh -w None -e Uv -m NM
# bash job1600.sh -w None -e vT -m NM
# bash job1600.sh -w xy -e vT -m NM
# bash job1600.sh -w xy -e UvT -m NM
# bash job1600.sh -w xy -e Uv -m NM
# bash job1600.sh -w xy -e UT -m NM
