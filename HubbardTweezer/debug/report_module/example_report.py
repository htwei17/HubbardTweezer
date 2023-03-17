# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:20:53 2022
@author: eibarragp
@edited by: Hao-Tian Wei

Example of how to use the report_module.py
"""

import reportIO as rep
import numpy as np

filename = 'haotian_example.ini'
report = rep.get_report(filename)

# Example of reading
print(rep.s(report, "Parameters", "geometry"))
print(rep.f(report, "Parameters", "nx"))
print(rep.b(report, "Parameters", "boolean_list"))
print(rep.b(report, "Parameters", "boolean_var"))
print(rep.a(report, "Parameters", "list_example"))
print(rep.a(report, "Output1", "qtraj"))

# Examples of writing 1
values = {"float_ex": -0.5, "bool_ex": False, "str_ex": "Test"}
rep.create_report(filename, "Output2", **values)

# Example of writing 2
A = np.zeros((3, 3, 3))
A[1] = 2 * np.eye(3, 3, k=1)
A[0] = -0.3 * np.eye(3, 3, k=-1)
A[2] = np.eye(3, 3)

B = np.arange(36).reshape((3, 3, 4))
values2 = {
    "array_1d": np.linspace(0, 1, 10),
    "array_2d": np.eye(3, 3),
    "array_3d": A,
    "array_3d2": B
}
rep.create_report(filename, "Output3", **values2)

# Let's read what we just wrote
report = rep.get_report(filename)
print(rep.s(report, "Output2", "str_ex"))
print(rep.f(report, "Output2", "float_ex"))
print(rep.b(report, "Output2", "bool_ex"))
print(rep.a(report, "Output3", "array_1d"))
print(rep.a(report, "Output3", "array_2d"))
print(rep.a(report, "Output3", "array_3d"))
print(rep.a(report, "Output3", "array_3d2"))
