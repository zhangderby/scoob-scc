from setuptools import setup, find_packages

VERSION = '0.1.0' 
DESCRIPTION = 'Package for running SCC on SCoOB.'
LONG_DESCRIPTION = 'Package for running SCC on SCoOB.'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="scoobscc",
        version=VERSION,
        author=["Kian Milani", "Kevin Derby",],
        author_email=["<kianmilani@arizona.edu>", "<derbyk@arizona.edu>",],
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[],
        keywords=['python', 'Coronagraph Instrument'],
        classifiers= [
            "Development Status :: Alpha-0.1.0",
            "Programming Language :: Python :: 3",
        ]
)

