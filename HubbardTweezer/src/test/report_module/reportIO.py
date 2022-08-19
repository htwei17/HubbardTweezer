# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:44:11 2022
@author: eibarragp
@edited by: Hao-Tian Wei

This is a module to read and write formatted input files for the optical
tweezers Hubbard parameters calculators
"""

from typing import ItemsView, Iterable
from configobj import ConfigObj
import numpy as np

global reportObj


############ GETTING THE REPORT ############
def get_report(report) -> ConfigObj:
    # Load the report
    reportObj = ConfigObj(report)
    return reportObj


############ READING THE REPORT ############
def f(report: ConfigObj, section: str, key=None, default=np.nan) -> float:
    # Return a float in a section from the already loaded report
    if (key == None):  # Formate "section:key" separated by ":" if key unspecified
        section, key = section.split(":")
    try:
        ret = float(report[section][key])
    except:  # If the key is not in the report
        try:
            if report[section][key] == 'None':  # If None is input
                print('Input is set to None.')
                ret = None
        except:
            ret = default
    return ret


def i(report: ConfigObj, section: str, key=None, default=-1) -> int:
    # Return an int in a section from the already loaded report
    if (key == None):  # Formate "section:key" separated by ":" if key unspecified
        section, key = section.split(":")
    try:
        ret = int(report[section][key])
    except:  # If the key is not in the report
        try:
            if report[section][key] == 'None':  # If None is input
                print('Input is set to None.')
                ret = None
        except:
            ret = default
    return ret


def s(report: ConfigObj, section: str, key=None, default='') -> str:
    # Return a string from the already loaded report
    if (key == None):  # Formate "section:key" separated by ":" if key unspecified
        section, key = section.split(":")
    try:
        ret = str(report[section][key])
    except:  # If the key is not in the report
        try:
            if report[section][key] == 'None':  # If None is input
                print('Input is set to None.')
                ret = None
        except:
            ret = default
    return ret


def a(report: ConfigObj,
      section: str,
      key=None,
      default=np.array([])) -> np.ndarray:
    # Return a numerical array from the already loaded report
    # NOTE: only works for 1, 2 and 3D arrays
    try:
        # For 1D arrays
        if isinstance(report[section][key], Iterable):
            ret = np.array(report[section][key]).astype(float)
        else:
            ret = np.array([report[section][key]]).astype(float)
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
            try:
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
                            current_element = float(
                                current_element.split("]")[0])
                        elif "[" in current_element:
                            current_element = float(
                                current_element.split("[")[1])
                        else:
                            pass
                        ret_matrix.append(float(current_element))
                    ec = int(num_elem / rb)
                    ret_matrix = np.array(ret_matrix).reshape(rb, ec)
                    ret.append(ret_matrix)
                ret = np.array(ret).reshape(rb, ec, len(ret_3))
            except:
                # If this is not any array listed above
                try:
                    if report[section][key] == 'None':  # If None is input
                        print('Input is set to None.')
                        ret = None
                except:
                    ret = default
    return ret


def b(report: ConfigObj, section: str, key=None, default=None) -> bool:
    # Return an array of booleans from the already loaded report
    if (key == None):  # Formate "section:key" separated by ":" if key unspecified
        section, key = section.split(":")
    try:
        dat = report[section][key]
        if isinstance(dat, str):
            ret = True if dat == 'True' else False
        else:
            ret = np.array([True if x == 'True' else False for x in dat])
    except:
        try:
            if report[section][key] == 'None':  # If None is input
                print('Input is set to None.')
                ret = None
        except:
            ret = default
    return ret


############ WRITE THE REPORT ############
def create_report(report, section: str, **kwargs) -> None:
    # Pass kwargs = {key:value, etc... }
    # Create a report section if it doesn't exist
    if isinstance(report, str):
        report = get_report(report)
        print('Report loaded')

    if not section in report.keys():
        # Create a new section if it doesn't exist
        report[section] = {}
        print("Created section: " + section)

    for key, value in kwargs.items():
        # if type(value) == list or np.ndarray:
        try:
            report[section][key] = value.tolist()
        except:
            report[section][key] = value
    report.write()
