
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


# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)
ext_modules = [
    Pybind11Extension(
        "ospa2._ospa2",           # <-- doit correspondre au module Python
        ["src/bindings.cpp", "src/ospa2.cpp"],
        include_dirs=[numpy.get_include(), "include", "/usr/include/eigen3"],
        cxx_std=17,
        # extra_compile_args=["-O3", "-fopenmp"],
        extra_compile_args=CXX_ARGS[platform.system()],
        extra_link_args=["-fopenmp"],
        define_macros=[("VERSION_INFO", __version__)],

    )
]
# ext_modules = [
#     Pybind11Extension(
#         "ospa2",
#         ["src/ospa2.cpp", "src/lapjv.cpp"],        # Example: passing in the version to the compiled code
#         #         include_dirs=[numpy.get_include(), "include", "/usr/include/eigen3"],

#         define_macros=[("VERSION_INFO", __version__)],
#     ),
# ]

setup(
    name="ospa2",
    version=__version__,
    author="Sylvain Corlay",
    author_email="sylvain.corlay@gmail.com",
    url="https://github.com/pybind/python_example",
    description="A test project using pybind11",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    packages=["ospa2"],

    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
)