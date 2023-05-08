import reportIO as rep
import numpy as np
import h5py
import glob
filename = 'output/samples/3D_4x4_square_None_neq_699.ini'
keys = ['V_offset', 'trap_centers','t_ij','V_i','U_i']
dat = {k:[] for k in keys}

for filename in glob.glob("output/samples/*.ini"):
    print(filename)
    report = rep.get_report(filename)

    dat['V_offset'].append(np.asarray([float(x) for x in report['Trap_Adjustments']['V_offset']]))
    dat['trap_centers'].append(rep.a(report,'Trap_Adjustments','trap_centers'))
    dat['t_ij'].append(rep.a(report, 'Singleband_Parameters','t_ij'))
    dat['V_i'].append(rep.a(report, 'Singleband_Parameters','V_i'))
    dat['U_i'].append(rep.a(report, 'Singleband_Parameters','U_i'))


with h5py.File("alldata.hdf5", 'w') as f:
    for k in keys:
        f[k] = np.asarray(dat[k])