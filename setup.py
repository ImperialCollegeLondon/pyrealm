import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

version = {}
with open("pyrealm/version.py") as fp:
    exec(fp.read(), version)

setuptools.setup(
    name="pyrealm",
    version=version['__version__'],
    author="David Orme",
    author_email="d.orme@imperial.ac.uk",
    description="Implementation of the pmodel in Python",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="",
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
    ],
)
