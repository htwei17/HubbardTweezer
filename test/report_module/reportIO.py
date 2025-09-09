# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:44:11 2022
@author: eibarragp
@edited by: Hao-Tian Wei

This is a module to read and write formatted input files for the optical
tweezers Hubbard parameters calculators
"""

from typing import Iterable
from configobj import ConfigObj
import json
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
    if key == None:  # Formate "section:key" separated by ":" if key unspecified
        section, key = section.split(":")
    try:
        ret = float(report[section][key])
    except:  # If the key is not in the report
        ret = default
    if ret == "None":
        print(f"Input item {key} is set to None.")
        ret = None
    return ret


def i(report: ConfigObj, section: str, key=None, default=-1) -> int:
    # Return an int in a section from the already loaded report
    if key == None:  # Formate "section:key" separated by ":" if key unspecified
        section, key = section.split(":")
    try:
        ret = int(report[section][key])
    except:  # If the key is not in the report
        ret = default
    if ret == "None":
        print(f"Input item {key} is set to None.")
        ret = None
    return ret


def s(report: ConfigObj, section: str, key=None, default="") -> str:
    # Return a string from the already loaded report
    if key == None:  # Formate "section:key" separated by ":" if key unspecified
        section, key = section.split(":")
    try:
        ret = str(report[section][key])
    except:  # If the key is not in the report
        ret = default
    # If None is input for string, this will not throw an ERROR
    if ret == "None":
        print(f"Input item {key} is set to None.")
        ret = None
    return ret


def recursion(target, dtype=float, output_arr=False) -> np.ndarray:
    if isinstance(target, str):
        # If the value is a single string (i.e., a string representing an array)
        # Try to parse the string as JSON to get a list
        # This should work for any number of dimensions
        readout = json.loads(target)
        if isinstance(readout, list):
            ret = np.array(readout).astype(dtype)
        else:
            # If the readout is not a list, return it as a single-element array
            ret = dtype(readout)
        return ret
    elif isinstance(target, Iterable):
        # If the value is a list of strings, convert it to an array
        # This is from data formatted as a = 1, 2, 3, 4, 5
        ret = [recursion(x, dtype=dtype) for x in target]
        if output_arr:
            ret = np.array(ret)
        return ret
    else:
        # If the value is not an Iterable
        raise TypeError(f"Input {target} is not an recursable Iterable.")


def a(
    report: ConfigObj, section: str, key=None, default=np.array([]), output_arr=False
) -> np.ndarray:
    try:
        target = report[section][key]
        ret = recursion(target, output_arr=output_arr)
    except:
        # If anything goes wrong, return the default value
        ret = default
    return ret


def b(report: ConfigObj, section: str, key=None, default=None) -> bool:
    # Return an array of booleans from the already loaded report
    if key == None:  # Formate "section:key" separated by ":" if key unspecified
        section, key = section.split(":")
    try:
        dat = report[section][key]
        if isinstance(dat, str):
            ret = True if dat == "True" else False
        else:
            ret = np.array([True if x == "True" else False for x in dat])
    except:
        ret = default
    if ret is None:
        print(f"Input item {key} is set to None.")
        ret = None
    return ret


############ WRITE THE REPORT ############
def create_report(report, section: str, **kwargs) -> None:
    # Pass kwargs = {key:value, etc... }
    # Create a report section if it doesn't exist
    if isinstance(report, str):
        report = get_report(report)
        print("Report loaded")

    if not section in report.keys():
        # Create a new section if it doesn't exist
        report[section] = {}
        print("Created section: " + section)

    for key, value in kwargs.items():
        # if type(value) == list or np.ndarray:
        try:
            # Convert the array to a list and then to a JSON string
            # This function dumps the array formatted as "[1, 2, 3, 4, 5]"
            report[section][key] = json.dumps(value.tolist())
        except:
            report[section][key] = value
    report.write()
