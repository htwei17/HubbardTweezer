# HubbatdTweezer

--------------------

![release](https://img.shields.io/github/v/release/Kvanti17/HubbardTweezer?color=green&include_prereleases)
[![DOI](https://zenodo.org/badge/519873426.svg)](https://zenodo.org/badge/latestdoi/519873426)

Hubbard parameter calculator for 1&2D optical tweezer array systems

Documentation [here](doc/manual.md).

Please cite the [paper](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.109.013318) if you wish to publish research work based on **HubbardTweezer**:

```bibtex
@article{PhysRevA.109.013318,
  title = {Hubbard parameters for programmable tweezer arrays},
  author = {Wei, Hao-Tian and Ibarra-Garc\'{\i}a-Padilla, Eduardo and Wall, Michael L. and Hazzard, Kaden R. A.},
  journal = {Phys. Rev. A},
  volume = {109},
  issue = {1},
  pages = {013318},
  numpages = {13},
  year = {2024},
  month = {Jan},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevA.109.013318},
  url = {https://link.aps.org/doi/10.1103/PhysRevA.109.013318}
}
```

## Features

1. Solve Hubbard parameters of 2d arbitrary finite geometry, including a set of lattice shapes and custom geometries
2. Equalize over all sites and engineer site-specific Hubbard parameters
3. Support both built-in Gaussian and customized DMD tweezer profiles

## Dependencies

* `scipy` along with `numpy`
* [`pymanopt`](https://github.com/pymanopt/pymanopt) which depends on [`torch`](https://github.com/pytorch/pytorch)
* [`opt_einsum`](https://github.com/dgasmith/opt_einsum)
* [`nlopt`](https://github.com/stevengj/nlopt)
* [`ortools`](https://github.com/google/or-tools)
* `configobj`
