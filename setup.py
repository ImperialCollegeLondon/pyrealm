import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

version = {}
with open("pypmodel/version.py") as fp:
    exec(fp.read(), version)

setuptools.setup(
    name="pypmodel",
    version=version['__version__'],
    author="David Orme",
    author_email="d.orme@imperial.ac.uk",
    description="Implementation of the pmodel in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=['pypmodel'],
    package_data={
        'pypmodel': ['data/params.yml'],
    },
    entry_points = {
            'console_scripts':
             ['']
    },
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
