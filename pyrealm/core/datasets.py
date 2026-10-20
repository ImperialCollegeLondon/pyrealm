"""The ``dataset`` module provides a download process for datasets from the
``pyrealm_build_data`` package. This package is only installed as part of the ``sdist``
package, so the files are accessible using :mod:`importlib.resources` for source
installations. However, we do not want to ship the ``pyrealm_build_data`` package as
part of the package binary because some of the files are quite large. This module
provides:

* The :meth:`get_pyrealm_data` function to access these datasets, either by
  using :mod:`importlib.resources` for use in development environments where
  ``pyrealm_build_data`` is present, or by using the :mod:`pooch` package to download
  requested files to a local cache.
* The :data:`DATASETS` object, which is a :class:`pooch.Pooch` instance that manages
  the available datasets.
* The ``pyrealm_data_registry.json`` file that provides a dictionary mapping the
  available datasets by relative path to their SHA256 hashes.
* The private :meth:`_populate_pooch_registry` function to generate
  ``pyrealm_data_registry.json`` from a local copy of ``pyrealm_build_data``.

The module contains an ``if __name__ == "__main__":`` section that allows the
:meth:`_populate_pooch_registry` function to be run using ``python -m
pyrealm.core.datasets``.
"""  # noqa: D205, D415

import hashlib
import json
import os
from importlib import resources

import pooch


def _populate_pooch_registry(
    dataset_filetypes: tuple[str, ...] = (".csv", ".nc", ".json"),
) -> None:
    """Generate a listing file of available pyrealm_build_data datasets.

    This function recursively searches the ``pyrealm_build_data`` package for files with
    suffixes matching the `dataset_filetypes` argument. It then writes the
    ``pyrealm_data_registry.json`` file into the :mod:`pyrealm.core` module, which
    contains a dictionary keying the file path of each dataset relative to the
    ``pyrealm_build_data`` root to the the SHA256 hash of the file.

    The command can be run manually using:

    .. code:: sh

        python -m  pyrealm.core.datasets

    Args:
        dataset_filetypes: A tuple of file suffixes for the types of file to be included
            in the registry.
    """

    # Get the pyrealm_build_data root directory
    pyrealm_build_data_dir = resources.files("pyrealm_build_data")

    # Recursively search the root directory for files matching the required filetypes
    with resources.as_file(pyrealm_build_data_dir) as root_path:
        data_files = (f for f in root_path.rglob("*") if f.suffix in dataset_filetypes)

        # Populate the registry data with the relative file path and SHA256 hash
        registry_data = {}

        for file in data_files:
            with open(file, "rb") as data:
                registry_data[str(file.relative_to(root_path))] = (
                    "sha256:" + hashlib.sha256(data.read()).hexdigest()
                )

    # Write the registry dictionary of SHA256 values keyed by relative path to
    # pyrealm_data_registry.json
    pyrealm_core_dir = resources.files("pyrealm.core")
    with resources.as_file(pyrealm_core_dir) as core_path:
        with open(core_path / "pyrealm_data_registry.json", "w") as registry:
            json.dump(registry_data, registry, indent=4)


DATASETS = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("pyrealm_data"),
    base_url="https://raw.githubusercontent.com/ImperialCollegeLondon/pyrealm/refs/heads/develop/pyrealm_build_data/",
    # The registry specifies the files that can be fetched
    registry=json.load(
        resources.files("pyrealm.core").joinpath("pyrealm_data_registry.json").open()
    ),
)
"""A :class:`pooch.Pooch` instance used to manage the available dataset files."""


def get_pyrealm_data(filepath: str, use_resources: bool = False) -> str:
    """Get a path to pyrealm_build_data datasets, possibly downloading the dataset.

    This function returns a path to a dataset provided in the :mod:`pyrealm_build_data`
    package. These datasets are not provided in the binary build of :mod:`pyrealm` so
    are only locally available from `sdist` installations or from within a ``git`` clone
    of the package repository. A list of available files is maintained in the
    ``pyrealm_data_registry.json`` file within the :mod:`pyrealm.core` module and
    managed through the :data:`DATASETS` file manager.

    The function runs in one of two ways:

    * If ``use_resources=True`` is used or the ``PYREALM_USE_LOCAL_DATA`` environment
      variable is set, then the function using :func:`importlib.resources.files` to
      provide a path to the local copy.
    * Otherwise, the function uses the :meth:`pooch.Pooch.fetch` method on the
      :data:`DATASETS` instance to download the requested file to a local cache and then
      returns the path to that file.

    The :mod:`pyrealm` package ``sphinx`` configuration for building documentation sets
    ``PYREALM_USE_LOCAL_DATA`` to ensure documentation builds use local data.

    Args:
        filepath: A string giving the path to the required dataset, relative to the root
            of the ``pyrealm_build_data`` package.
        use_resources: A boolean toggle to switch to providing a path to an existing
            local installation of the ``pyrealm_build_data`` package, rather than
            downloading the requested file.
    """
    if filepath not in DATASETS.registry:
        raise ValueError(f"Unknown file: {filepath}")

    if use_resources or ("PYREALM_USE_LOCAL_DATA" in os.environ):
        return str(resources.files("pyrealm_build_data") / filepath)

    return DATASETS.fetch(filepath)


# If the module is executed as a script, populate the registry.
if __name__ == "__main__":
    # If the module is executed as a script, populate the registry.
    _populate_pooch_registry()
