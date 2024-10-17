"""The ``pyrealm`` repository includes both the ``pyrealm`` package and the
``pyrealm_build_data`` package. The ``pyrealm_build_data`` package contains datasets
that are used in the ``pyrealm`` build and testing process. This includes:

* Example datasets that are used in the package documentation, such as simple spatial
  datasets for showing the use of the P Model.
* "Golden" datasets for regression testing ``pyrealm`` implementations against the
  outputs of other implementations. These datasets will include a set of input data and
  then output predictions from other implementations.
* Datasets for providing profiling of ``pyrealm`` code and for benchmarking new versions
  of the package code against earlier implementations to check for performance issues.

The package is organised into submodules that reflect the data use or previous
implementation.

Note that ``pyrealm_build_data`` is a source distribution only (``sdist``) component of
``pyrealm``, so is not included in binary distributions (``wheel``) that are typically
installed by end users. This means that files in ``pyrealm_build_data`` are not
available if a user has simply used ``pip install pyrealm``: please *do not* use
``pyrealm_build_data`` within the main ``pyrealm`` code.
"""  # noqa: D205, D415
