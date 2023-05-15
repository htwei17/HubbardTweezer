#!/bin/zsh

# This script converts the the output INI file to having the json multidimensional string
# This is done by using the python script convert.py

for file in $(find $1 -name "*.ini"); do
    python $(dirname "$0")/convert.py "$file"
done
