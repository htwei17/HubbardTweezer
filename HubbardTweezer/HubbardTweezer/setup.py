def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration("HubbardTweezer", parent_package, top_path)
    config.version = "dev"
    config.add_subpackage("DVR")
    config.add_subpackage("Hubbard")
    config.add_subpackage("tools")
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(
        **configuration(top_path="").todict(),
        description="HubbardTweezer: Hubbard parameter calculator for 1&2D optical tweezer array systems",
        author="Hao-Tian Wei",
        author_email="htwei@rice.edu",
        url="https://github.com/htwei17/HubbardTweezer.git",
        install_requires=[
            "numpy",
            "scipy",
            "nlopt",
            "matplotlib",
            "torch",
            "pymanopt",
            "networkx",
            "configobj",
            "opt_einsum",
            "pympler",
            "h5py",
            "ortools",
        ],
    )
