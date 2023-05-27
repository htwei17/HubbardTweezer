#!/bin/bash

if [ -s 3D_3x3_Lieb_neq.ini ]; then
    echo "this is not empty"
else
    echo "this is empty" >>3D_3x3_Lieb_neq.ini
fi

if [ -s empty.txt ]; then
    echo "the non-existing file is detected."
else
    echo "the non-existing file is detected to be empty."
fi
