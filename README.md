# HubbardTweezer

-----------------

![release](https://img.shields.io/github/v/release/Kvanti17/HubbardTweezer?color=green&include_prereleases)
[![DOI](https://zenodo.org/badge/519873426.svg)](https://zenodo.org/badge/latestdoi/519873426)

Hubbard parameter calculator for 1&2D optical tweezer array systems

Documentation [here](doc/mannual.md).

Please cite the [paper]() if you wish to publish research work based on HubbardTweezer:

```bibtex
@article{xxx}
```

## Features

1. Dynamics of on-off strobe potential
2. Solve Hubbard parameters of 2d arbitrary geometry
3. Equalize Hubbard parameters over all sites

## Dependencies

* [`pymanopt`](https://github.com/pymanopt/pymanopt) which depends on [`torch`](https://github.com/pytorch/pytorch)
* [`networkx`](https://github.com/networkx/networkx) which depends on `matplotlib`
* [`opt_einsum`](https://github.com/dgasmith/opt_einsum)
* `scipy` along with `numpy`
* `ortools`
* `configobj`
* `pympler`
* `h5py`
