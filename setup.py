
import numpy

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import platform

__version__ = "0.0.1"


UNIX_CXXFLAGS = [
    "-std=c++17",
    "-mavx2",
    "-ftree-vectorize",
    # GCP N2
    "-march=haswell",
    "-maes",
    "-mno-pku",
    "-mno-sgx",
    "--param", "l1-cache-line-size=64",
    "--param", "l1-cache-size=32",
    "--param", "l2-cache-size=33792",
]

CXX_ARGS = {
    # "Darwin": [*UNIX_CXXFLAGS],  not supported anymore due to M1, PRs welcome
    "Linux": ["-fopenmp", *UNIX_CXXFLAGS, "-mabm"],
    "Windows": ["/openmp", "/std:c++latest", "/arch:AVX2"],
}



ext_modules = [
    Pybind11Extension(
        "PyOSPA2._ospa2",           
        ["src/bindings.cpp", "src/ospa2.cpp"],
        include_dirs=[numpy.get_include(), "include", "/usr/include/eigen3"],
        cxx_std=17,
        # extra_compile_args=["-O3", "-fopenmp"],
        extra_compile_args=CXX_ARGS[platform.system()],
        extra_link_args=["-fopenmp"],
        define_macros=[("VERSION_INFO", __version__)],

    )
]


setup(
    name="PyOSPA2",
    version=__version__,
    author="Alexandre Demarquet",
    author_email="alex.demarquet@gmail.com",
    description="A simple implementation of the OSPA² metric for multi-object tracking evaluation.",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    packages=["PyOSPA2"],

    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
)