#!/bin/bash
# Loop over all files in the directory $1
# $1 is the first argument to the script

for file in $(ls "$1"); do
    echo "Processing '$1'$file"
    python -O -u src/Hubbard_exe.py "$1"$file
done
