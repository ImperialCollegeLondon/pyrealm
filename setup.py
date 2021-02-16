import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

version = {}
with open("pyrealm/version.py") as fp:
    exec(fp.read(), version)

setuptools.setup(
    name="pyrealm",
    version=version['__version__'],
    author="David Orme",
    author_email="d.orme@imperial.ac.uk",
    description="Python implementations of REALM models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pyrealm.readthedocs.io/",
    packages=setuptools.find_packages(),
    package_data={
        'pyrealm': ['data/*'],
    },
    entry_points={
            'console_scripts':
             ['']
    },
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Development Status :: 3 - Alpha"
    ],
)
