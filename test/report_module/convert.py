from configobj import ConfigObj
import numpy as np
import json
from typing import Iterable

global reportObj


def a(report: ConfigObj, section: str, key=None, default=np.array([])) -> np.ndarray:
    # Return a numerical array from the already loaded report
    # NOTE: only works for 1, 2 and 3D arrays
    try:
        # For 1D arrays
        # In case somebody writes "a = 4" instead of "a = 4,"
        # s.t. the object is a float instead of a string
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
                    ret_2[i].lstrip("[").rstrip("]").split(",")
                ).astype(float)
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
            except:
                # If this is not any array listed above
                ret = default
    return ret


def get_report(report) -> ConfigObj:
    # Load the report
    reportObj = ConfigObj(report)
    return reportObj


def convert(report):
    for section in report:
        if section not in ["Parameters"]:
            for key in report[section]:
                ret = a(report, section, key)
                if ret.size > 0:
                    report[section][key] = json.dumps(ret.tolist())
    return report


# script.py
import sys

if __name__ == "__main__":
    first_arg = sys.argv[1]
    report = get_report(first_arg)
    report = convert(report)
    report.write()
