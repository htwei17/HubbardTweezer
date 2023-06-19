# HubbardTweezer

-----------------

![release](https://img.shields.io/github/v/release/Kvanti17/HubbardTweezer?color=green&include_prereleases)
[![DOI](https://zenodo.org/badge/519873426.svg)](https://zenodo.org/badge/latestdoi/519873426)

Hubbard parameter calculator for 1&2D optical tweezer array systems

Documentation [here](doc/mannual.md).

Please cite the [paper](https://arxiv.org/abs/2306.03019) if you wish to publish research work based on HubbardTweezer:

```bibtex
@ARTICLE{2023arXiv230603019W,
       author = {{Wei}, Hao-Tian and {Ibarra-Garc{\'\i}a-Padilla}, Eduardo and {Wall}, Michael L. and {Hazzard}, Kaden R.~A.},
        title = "{Hubbard parameters for programmable tweezer arrays}",
      journal = {arXiv e-prints},
     keywords = {Condensed Matter - Quantum Gases, Physics - Atomic Physics, Quantum Physics},
         year = 2023,
        month = jun,
          eid = {arXiv:2306.03019},
        pages = {arXiv:2306.03019},
          doi = {10.48550/arXiv.2306.03019},
archivePrefix = {arXiv},
       eprint = {2306.03019},
 primaryClass = {cond-mat.quant-gas},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230603019W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Features

1. Solve Hubbard parameters of 2d arbitrary finite lattice geometry
2. Equalize Hubbard parameters over all sites

## Dependencies

* `scipy` along with `numpy`
* [`pymanopt`](https://github.com/pymanopt/pymanopt) which depends on [`torch`](https://github.com/pytorch/pytorch)
* [`opt_einsum`](https://github.com/dgasmith/opt_einsum)
* [`nlopt`](https://github.com/stevengj/nlopt)
* [`ortools`](https://github.com/google/or-tools)
* `configobj`
