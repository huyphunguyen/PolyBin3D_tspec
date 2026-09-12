from setuptools import setup

setup(
    name="polybin3d_tspec",
    version="0.1.0",
    packages=["polybin3d_tspec", "polybin3d_tspec.cython"],
    package_dir={"polybin3d_tspec": "."},
    package_data={"polybin3d_tspec.cython": ["*.pyx"]},
)
