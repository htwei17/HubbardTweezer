import argparse
from os.path import exists, splitext

import h5py
import numpy as np
import numpy.linalg as la

from ..tools import reportIO as rep
from .core import interaction
from .equalizer import HubbardEqualizer, str_to_flags
from .io import write_singleband, write_trap_params, write_wannier
from .lattice import Lattice

HELP_EPILOG = """
The program reads input sections from the supplied INI file and appends
results to that same file. Duplicate keys within a section raise an error.

Input Sections
--------------

[DVR_Parameters]
- N (default 20)
- L0 (default 3,3,7.2)
- DVR_dimension (default 1)
- Optional: sparse (default True), DVR_symmetry (default True)

[Lattice_Parameters]
- shape (default square; supports square, Lieb, triangular, honeycomb,
  defecthoneycomb, kagome, ring, zigzag, Penrose, custom)
- lattice_size (default 4,)
- lattice_const (default 1520,1690 nm)
- lattice_symmetry (default True)
- potential_model (default Gaussian)
  * when custom, provide both custom_potential_grid and custom_potential_value
  * when shape=custom, supply site_locations and optional bond_links

[Trap_Parameters]
- scattering_length (default 1000 a0)
- V0 (default 104.52 kHz)
- waist (default 1000,1000 nm)
- atom_mass (default 6.015122 amu)
- zR (default None, derived from waist and wavelength)
- laser_wavelength (default 780 nm)
- average (default 1)

[Hubbard_Settings]
- band (default 1)
- zero_average_V (default True)
- calculate_U (default True)
- Nintgrl_grid (default 200)
- offdiagonal_U (default False)

[Equalization_Parameters]
- equalize (default False)
- equalize_item (default vT; lowercase updates targets every iteration,
  uppercase locks to the initial guess)
- balance_V0 (default False)
- waist_direction (default None)
- method (default trf; supports dogbox, Nelder-Mead/NM, Powell, bfgs,
  L-BFGS-B, cobyla, SLSQP, and NLopt methods bobyqa/praxis/subplex/direct/crs2)
- no_bounds (default False)
- random_initial_guess (default False)
- ghost_sites (default False) and ghost_penalty (default 1,1)
- Targets: U_target, t_target, V_target (all default None)
- Scaling: scale_factor (default None)

[Verbosity]
- write_log (default False)
- verbosity (default 0)
- output_lowest_wannier (default False)

[Equalization_Result]
- Optional: x (initial guess) or simplex (for Nelder-Mead)
- Optional: U_over_t
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Hubbard parameters from an INI report.",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", help="Path to the input INI file.")
    return parser


def load_report(input_file: str):
    if not exists(input_file):
        raise FileNotFoundError(f"Hubbard_exe: input file not found: {input_file}")
    return rep.get_report(input_file)


def _read_custom_potential(report, shape: str, model: str):
    nodes = None
    links = None
    custom_potential = None

    if model in ["Gaussian", "optical_lattice"]:
        if model == "optical_lattice" and shape != "square":
            raise ValueError(
                "optical_lattice model only supports square lattice shape; use Gaussian for other shapes."
            )
        if shape == "custom":
            nodes = rep.a(report, "Lattice_Parameters", "site_locations", None)
            links = rep.a(report, "Lattice_Parameters", "bond_links", None).astype(int)
    elif model == "custom":
        custom_potential_grid = rep.a(
            report, "Lattice_Parameters", "custom_potential_grid", None
        )
        custom_potential_value = rep.a(
            report, "Lattice_Parameters", "custom_potential_value", None
        )
        if custom_potential_grid is not None and custom_potential_value is not None:
            custom_potential = (custom_potential_grid, custom_potential_value)
        else:
            print(
                "Custom potential grid or value not provided, using default potential."
            )

    return nodes, links, custom_potential


def _read_target_values(report):
    Utarget = rep.a(
        report, "Equalization_Parameters", "U_target", None, output_arr=True
    )
    t_target = rep.a(report, "Equalization_Parameters", "t_target", None)
    Vtarget = rep.a(
        report, "Equalization_Parameters", "V_target", None, output_arr=True
    )

    if not any(value is not None for value in (Utarget, t_target, Vtarget)):
        return None

    if t_target is None:
        tx_target, ty_target = None, None
    elif len(t_target) == 1:
        tx_target, ty_target = t_target[0], None
    elif len(t_target) == 2:
        tx_target, ty_target = t_target
    else:
        raise ValueError(
            "t_target must be a single value/array or a pair of values/arrays for tx and ty."
        )
    return (Vtarget, Utarget, tx_target, ty_target)


def _read_initial_guess(report, method: str):
    if method == "Nelder-Mead":
        return rep.a(
            report,
            "Equalization_Result",
            "simplex",
            rep.a(report, "Equalization_Result", "x", None),
        )
    return rep.a(report, "Equalization_Result", "x", None)


def read_input_sections(report) -> dict:
    dvr = {
        "N": rep.i(report, "DVR_Parameters", "N", 20),
        "L0": rep.a(
            report, "DVR_Parameters", "L0", np.array([3, 3, 7.2]), output_arr=True
        ),
        "dimension": rep.i(report, "DVR_Parameters", "DVR_dimension", 1),
        "sparse": rep.b(report, "DVR_Parameters", "sparse", True),
        "symmetry": rep.b(report, "DVR_Parameters", "DVR_symmetry", True),
    }

    shape = rep.s(report, "Lattice_Parameters", "shape", "square")
    model = rep.s(report, "Lattice_Parameters", "potential_model", "Gaussian")
    nodes, links, custom_potential = _read_custom_potential(report, shape, model)
    lattice = {
        "shape": shape,
        "lattice_symmetry": rep.b(
            report, "Lattice_Parameters", "lattice_symmetry", True
        ),
        "lattice_const": tuple(
            rep.a(report, "Lattice_Parameters", "lattice_const", [1520, 1690])
        ),
        "lattice_size": rep.a(
            report, "Lattice_Parameters", "lattice_size", np.array([4]), output_arr=True
        ).astype(int),
        "nodes": nodes,
        "links": links,
        "model": model,
        "custom_potential": custom_potential,
    }

    trap = {
        "scattering_length": rep.f(
            report, "Trap_Parameters", "scattering_length", 1000
        ),
        "V0": rep.f(report, "Trap_Parameters", "V0", 104.52),
        "waist": rep.a(
            report, "Trap_Parameters", "waist", np.array([1000, 1000]), output_arr=True
        ),
        "atom_mass": rep.f(report, "Trap_Parameters", "atom_mass", 6.015122),
        "zR": rep.f(report, "Trap_Parameters", "zR", None),
        "laser_wavelength": rep.f(report, "Trap_Parameters", "laser_wavelength", 780),
        "average": rep.f(report, "Trap_Parameters", "average", 1),
    }

    hubbard = {
        "band": rep.i(report, "Hubbard_Settings", "band", 1),
        "zero_average_V": rep.b(report, "Hubbard_Settings", "zero_average_V", True),
        "calculate_U": rep.b(report, "Hubbard_Settings", "calculate_U", True),
        "Nintgrl_grid": rep.i(report, "Hubbard_Settings", "Nintgrl_grid", 200),
        "offdiag_U": rep.b(report, "Hubbard_Settings", "offdiagonal_U", False),
        "U_over_t": rep.f(report, "Equalization_Result", "U_over_t", None),
    }

    method = rep.s(report, "Equalization_Parameters", "method", "trf")
    method = "Nelder-Mead" if method == "NM" else method
    equalization = {
        "equalize": rep.b(report, "Equalization_Parameters", "equalize", False),
        "equalize_item": rep.s(
            report, "Equalization_Parameters", "equalize_item", "vT"
        ),
        "balance_V0": rep.b(report, "Equalization_Parameters", "balance_V0", False),
        "waist_direction": rep.s(
            report, "Equalization_Parameters", "waist_direction", None
        ),
        "method": method,
        "no_bounds": rep.b(report, "Equalization_Parameters", "no_bounds", False),
        "ghost_sites": rep.b(report, "Equalization_Parameters", "ghost_sites", False),
        "ghost_penalty": rep.a(
            report,
            "Equalization_Parameters",
            "ghost_penalty",
            np.array([1, 1]),
            output_arr=True,
        ),
        "random_initial_guess": rep.b(
            report, "Equalization_Parameters", "random_initial_guess", False
        ),
        "target_values": _read_target_values(report),
        "scale_factor": rep.f(report, "Equalization_Parameters", "scale_factor", None),
        "x0": _read_initial_guess(report, method),
    }

    if equalization["equalize"] and lattice["model"] == "custom":
        raise ValueError("Equalization not supported for custom potential profiles.")

    verbosity = {
        "write_log": rep.b(report, "Verbosity", "write_log", False),
        "verbosity": rep.i(report, "Verbosity", "verbosity", 0),
        "output_lowest_wannier": rep.b(
            report, "Verbosity", "output_lowest_wannier", False
        ),
        "save_format": rep.s(report, "Verbosity", "save_format", "ini"),
    }

    return {
        "dvr": dvr,
        "lattice": lattice,
        "trap": trap,
        "hubbard": hubbard,
        "equalization": equalization,
        "verbosity": verbosity,
    }


def build_lattice(inputs: dict) -> Lattice:
    lattice = inputs["lattice"]
    equalization = inputs["equalization"]
    return Lattice(
        shape=lattice["shape"],
        lattice_symmetry=lattice["lattice_symmetry"],
        lattice=lattice["lattice_size"],
        lc=lattice["lattice_const"],
        nodes=lattice["nodes"],
        links=lattice["links"],
        isotropic=False,
        ghost=equalization["ghost_sites"],
        ghost_penalty=equalization["ghost_penalty"],
    )


def build_hubbard_equalizer(report, inputs: dict, lattice: Lattice) -> HubbardEqualizer:
    dvr = inputs["dvr"]
    lattice_params = inputs["lattice"]
    trap = inputs["trap"]
    hubbard = inputs["hubbard"]
    equalization = inputs["equalization"]
    verbosity = inputs["verbosity"]

    return HubbardEqualizer(
        dvr["N"],
        R0=dvr["L0"],
        dim=dvr["dimension"],
        lattice=lattice,
        custom_potential=lattice_params["custom_potential"],
        ascatt=trap["scattering_length"],
        band=hubbard["band"],
        avg=trap["average"],
        model=lattice_params["model"],
        trap=(trap["V0"], trap["waist"]),
        atom=trap["atom_mass"],
        laser=trap["laser_wavelength"],
        zR=trap["zR"],
        variable_waist=equalization["waist_direction"],
        sparse=dvr["sparse"],
        zero_avgV=hubbard["zero_average_V"],
        equalize=equalization["equalize"],
        eqitem=equalization["equalize_item"],
        balance_V0=equalization["balance_V0"],
        Ut=hubbard["U_over_t"],
        target_values=equalization["target_values"],
        Nintgrl_grid=hubbard["Nintgrl_grid"],
        random=equalization["random_initial_guess"],
        x0=equalization["x0"],
        scale_factor=equalization["scale_factor"],
        eqmethod=equalization["method"],
        nobounds=equalization["no_bounds"],
        symmetry=dvr["symmetry"],
        iofile=report,
        write_log=verbosity["write_log"],
        verbosity=verbosity["verbosity"],
    )


def write_h5_output(
    input_file: str, equalizer: HubbardEqualizer, total_cost: float
) -> None:
    out_file = splitext(input_file)[0] + ".h5"
    tij = abs(equalizer.A)
    tij += np.diag(np.diag(equalizer.A) - np.diag(tij))
    data = {
        "t_ij": tij,
        "U_i": equalizer.U,
        "V_offset": equalizer.Voff,
        "trap_centers": equalizer.lattice.trap_centers,
        "wf_centers": equalizer.wf_centers,
        "wf_cost": equalizer.wf_cost,
        "total_cost_func": total_cost,
    }
    with h5py.File(out_file, "w") as handle:
        print(f"Writing to h5 file {out_file} ...")
        for key, value in data.items():
            handle[key] = np.asarray(value)
        print("Done!")


def write_multiband_output(
    report, equalizer: HubbardEqualizer, eig_sol, calculate_U: bool
) -> None:
    if equalizer.bands <= 1:
        return

    maskedA, W, wf_centers, wf_costs = equalizer.multiband_WF(*eig_sol)
    values = {}
    for i in range(equalizer.bands):
        Vi = np.real(np.diag(maskedA[i]))
        tij = abs(np.real(maskedA[i] - np.diag(Vi)))
        values[f"t_{i+1}_ij"] = tij
        values[f"V_{i+1}_i"] = Vi
        values[f"wf_{i+1}_centers"] = wf_centers[i]
        values[f"wf_{i+1}_cost"] = wf_costs[i]

    if calculate_U:
        U = interaction(equalizer, W, *eig_sol[1:])
        for i in range(equalizer.bands):
            for j in range(equalizer.bands):
                values[f"U_{i+1}{j+1}_i"] = U[i, j]

    rep.create_report(report, "Multiband_Parameters", **values)


def run_report(report, input_file: str) -> HubbardEqualizer:
    inputs = read_input_sections(report)
    equalizer = build_hubbard_equalizer(report, inputs, build_lattice(inputs))

    equalization = inputs["equalization"]
    hubbard = inputs["hubbard"]
    verbosity = inputs["verbosity"]

    if not equalization["equalize"]:
        equalizer.Voff = rep.a(report, "Trap_Adjustments", "V_offset", equalizer.Voff)

    eig_sol = equalizer.eigen_basis()
    _, _, WF = equalizer.singleband_Hubbard(u=hubbard["calculate_U"], eig_sol=eig_sol)
    maskedA = equalizer.lattice.ghost.mask_quantity(equalizer.A)
    maskedU = (
        equalizer.lattice.ghost.mask_quantity(equalizer.U)
        if hubbard["calculate_U"]
        else None
    )
    links = equalizer.xy_links(equalizer.lattice.ghost.links)

    nnt = equalizer.lattice.nn_tunneling(maskedA)
    if equalizer.sf is None:
        equalizer.sf, _ = equalizer.txy_target(nnt, links, np.mean)
    if equalizer.verbosity > 1:
        print(f"scale_factor = {equalizer.sf}")
        print(f"V = {np.diag(equalizer.A)}")
        print(f"t = {abs(equalizer.lattice.nn_tunneling(equalizer.A))}")
        print(f"U = {equalizer.U}")

    write_singleband(report, equalizer)
    write_wannier(
        report,
        equalizer,
        verbosity["output_lowest_wannier"],
        eig_sol[1][0],
        eig_sol[2][0],
        WF,
    )
    if equalizer.bands == 1 and hubbard["calculate_U"] and hubbard["offdiag_U"]:
        print("Singleband off-diagonal U calculation.")
        _, W, _, _ = equalizer.multiband_WF(*eig_sol)
        U = interaction(equalizer, W, *eig_sol[1:], onsite=False)[0][0]
        rep.create_report(report, "Singleband_Parameters", U_ijkl=U)
    write_trap_params(report, equalizer)

    eqitem = (
        "uvt"
        if equalization["equalize_item"] == "neq"
        else equalization["equalize_item"]
    )
    u_flag, t_flag, v_flag, _, _, _ = str_to_flags(eqitem)
    weights = np.array([u_flag, t_flag, v_flag])

    target_values = equalization["target_values"]
    if target_values is None:
        Vtarget = np.mean(np.real(np.diag(maskedA)))
        tTarget = equalizer.txy_target(nnt, links, np.mean)
    else:
        Vtarget, Utarget, txTarget, tyTarget = target_values
        tTarget = [txTarget, tyTarget]

    ct = equalizer.t_cost_func(maskedA, links, tTarget, equalizer.sf)
    cv = equalizer.v_cost_func(maskedA, Vtarget, equalizer.sf)
    if hubbard["calculate_U"]:
        if target_values is None:
            Utarget = np.mean(maskedU)
        cu = equalizer.u_cost_func(maskedU, Utarget, equalizer.sf)
    else:
        Utarget = 0
        cu = 0
    cvec = np.array((cu, ct, cv))
    total = weights @ cvec
    cvec = np.sqrt(cvec)
    fval = np.sqrt(total)
    ctot = la.norm(cvec)
    equalizer.eqinfo["sf"] = equalizer.sf
    equalizer.eqinfo["Ut"] = np.max(Utarget) / np.min(tTarget[0])

    if equalization["equalize"]:
        equalizer.eqinfo.update_cost(cvec, fval, ctot)
    else:
        v0, _ = equalizer.init_v0_and_bound(random=False)
        equalizer.eqinfo.create_log(v0, (Vtarget, Utarget, *tTarget))
        equalizer.eqinfo.update_cost(cvec, fval, ctot)
        equalizer.eqinfo["success"] = False
        equalizer.eqinfo["exit_status"] = -1
        equalizer.eqinfo["termination_reason"] = "Not equalized"
    equalizer.eqinfo.write_equalization(report, write_log=verbosity["write_log"])

    if verbosity["save_format"] == "h5":
        write_h5_output(input_file, equalizer, ctot)

    write_multiband_output(report, equalizer, eig_sol, hubbard["calculate_U"])
    return equalizer


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = load_report(args.input_file)
        run_report(report, args.input_file)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
