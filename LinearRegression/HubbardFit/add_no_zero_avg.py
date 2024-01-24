# For all the files in the folder, insert a section "[Hubbard_Settings]" and a line "zero_average_V = False" from Line 15 to 16.

import reportIO as rep
import numpy as np
import h5py
import glob
import os
import configparser

# path = "/Users/nottforestfc/Library/CloudStorage/OneDrive-RiceUniversity/Documents/Research/Hubbard Tweezer Parameters/output/LinearRegression/samples_new_test/"
path = os.path.abspath(
    "../../OneDrive - Rice University/Documents/Research/Hubbard Tweezer Parameters/output/LinearRegression/samples_new/"
)


def insert_hubbard_settings(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".ini"):
            file_path = os.path.join(folder_path, file_name)

            # Read the file content
            with open(file_path, "r") as file:
                lines = file.readlines()

            # Check if file has at least 15 lines
            if len(lines) >= 15:
                if lines[14] != "[Hubbard_Settings]\n":
                    # Insert the required section and line
                    lines.insert(14, "[Hubbard_Settings]\n")
                    lines.insert(15, "zero_average_V = False\n")
                if lines[16] == "[Hubbard_Settings]\n":
                    # Remove the repeated section and line
                    lines.pop(16)
                    lines.pop(16)

                # Write the modified content back to the file
                with open(file_path, "w") as file:
                    file.writelines(lines)


# Usage
insert_hubbard_settings(path)
