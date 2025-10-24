import sys

from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

pkg_name = "H2iVDI"
__version__ = "0.1"

__extra_compile_args__ = ["-O3"]
#__extra_compile_args__ = ["-g", "-DDEBUG"]

# TODO restrict ext_modules with only extension methods
ext_modules = [
    Pybind11Extension(pkg_name + ".H2iVDI_ext",
        ["src/geometry.cpp",
         "src/standard_step.cpp",
         "src/h2ivdi_ext.cpp"],
         extra_compile_args=__extra_compile_args__,
        #  extra_compile_args=["-g"],
        #  extra_compile_args=["-g", "-D__CHECK_INTERNAL_ERRORS"],
        # Example: passing in the version to the compiled code
        #define_macros = [('VERSION_INFO', __version__)],
        ),
]

setup(
    name=pkg_name,
    version=__version__,
    author="Kevin Larnier",
    author_email="klarnier@gmail.com",
    description="H2iVDI (Hybrid Hierarchical Variational Discharge Inference) discharge algorithm",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    packages=find_packages(),
    python_requires=">=3.9",
)
