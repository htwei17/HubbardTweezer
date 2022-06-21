# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:44:11 2022
@author: eibarragp
@edited by: Hao-Tian Wei

This is a module to read and write formatted input files for the optical
tweezers Hubbard parameters calculators
"""

from configobj import ConfigObj
import numpy as np
global reportObj


############ GETTING THE REPORT ############
def get_report(report) -> ConfigObj:
    # Load the report
    reportObj = ConfigObj(report)
    return reportObj


############ READING THE REPORT ############
def f(report: ConfigObj, section: str, key=None) -> float:
    # Return a float in a section from the already loaded report
    if (key == None):
        section, key = section.split(":")
    try:
        ret = float(report[section][key])
    except:
        ret = np.nan
    return ret


def s(report: ConfigObj, section: str, key=None) -> str:
    # Return a string from the already loaded report
    if (key == None):
        section, key = section.split(":")
    try:
        ret = str(report[section][key])
    except:
        ret = "Not a string"
    return ret


def a(report: ConfigObj, section: str, key=None) -> np.ndarray:
    # Return an array from the already loaded report
    try:
        # For 1D arrays
        ret = np.array(report[section][key]).astype(float)
    except:
        try:
            # For 2D arrays
            ret_2 = report[section][key]
            ret = []
            for i in range(0, len(ret_2)):
                current_row = np.array(
                    ret_2[i].lstrip("[").rstrip("]").split(",")).astype(float)
                ret.append(current_row)
            ret = np.vstack(ret)
        except:
            # For 3D arrays
            ret_3 = report[section][key]
            ret_2 = []
            ret = []
            for i in range(0, len(ret_3)):
                ret_2 = ret_3[i].lstrip("[").rstrip("]").split(",")
                ret_matrix = []
                rb = 1
                num_elem = len(ret_2)
                for j in range(0, num_elem):
                    current_element = ret_2[j]
                    if "]" in current_element:
                        rb += 1
                        current_element = float(current_element.split("]")[0])
                    elif "[" in current_element:
                        current_element = float(current_element.split("[")[1])
                    else:
                        pass
                    ret_matrix.append(float(current_element))
                ec = int(num_elem / rb)
                ret_matrix = np.array(ret_matrix).reshape(rb, ec)
                ret.append(ret_matrix)
            ret = np.array(ret).reshape(rb, ec, len(ret_3))
    return ret


def b(report: ConfigObj, section: str, key=None) -> bool:
    # Return an array of booleans from the already loaded report
    dat = report[section][key]
    if isinstance(dat, str):
        ret = bool(dat)
    else:
        try:
            ret = np.array([True if x == 'True' else False for x in dat])
        except:
            ret = None
    return ret


############ WRITE THE REPORT ############
def create_report(report, section, **kwargs) -> None:
    # Pass kwargs = {key:value, etc... }
    # Create a report section if it doesn't exist
    report = get_report(report)

    if not section in report.keys():
        report[section] = {}

    for key, value in kwargs.items():
        #if type(value) == list or np.ndarray:
        try:
            report[section][key] = value.tolist()
        except:
            report[section][key] = value
    report.write()
