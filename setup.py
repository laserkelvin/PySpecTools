from setuptools import setup

setup(
    name="pyspectools",
    version="0.1",
    description="A set of Python tools/routines for spectroscopy",
    author="Kelvin Lee",
    packages=["pyspectools"],
    include_package_data=True,
    author_email="kin_long_kelvin.lee@cfa.harvard.edu",
    install_requires=[
            "numpy",
            "pandas",
            "scipy",
            "colorlover",
            "matplotlib"
    ]
)
