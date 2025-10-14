"""The ``dataset`` module provides a download process for datasets from the
``pyrealm_build_data`` package. This package is only installed as part of the ``sdist``
package, so the files are accessible using importlib for source installations, but is
not shipped with the binary package on PyPI. This module provides the
``get_pyrealm_data`` function to access example datasets.
"""  # noqa: D205

import hashlib
import os
from importlib import resources

import pooch


def _populate_pooch_registry() -> None:
    pyrealm_build_data_dir = resources.files("pyrealm_build_data")

    with resources.as_file(pyrealm_build_data_dir) as root_path:
        data_files = (
            f for f in root_path.rglob("*") if f.suffix in {".csv", ".nc", ".json"}
        )

        registry_data = []

        for file in data_files:
            with open(file, "rb") as data:
                registry_data.append(
                    (
                        str(file.relative_to(root_path)),
                        "sha256:" + hashlib.sha256(data.read()).hexdigest(),
                    )
                )

    pyrealm_core_dir = resources.files("pyrealm.core")
    with resources.as_file(pyrealm_core_dir) as core_path:
        with open(core_path / "pyrealm_data_registry.txt", "w") as registry:
            registry.writelines(
                [f"{file_path} {file_hash}\n" for file_path, file_hash in registry_data]
            )


DATASETS = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("pyrealm_data"),
    base_url="https://raw.githubusercontent.com/ImperialCollegeLondon/pyrealm/refs/heads/develop/pyrealm_build_data/",
    # The registry specifies the files that can be fetched
    # TODO: replace None values with file hashes. Maybe make this automatically for
    # _all_ build data files?
    registry=None,
)

DATASETS.load_registry(resources.files("pyrealm.core") / "pyrealm_data_registry.txt")


def get_pyrealm_data(filepath: str, use_resources: bool = False) -> str:
    """It is _mystifyingly_ hard to get resources.files() Traversable as a Path."""
    if filepath not in DATASETS.registry:
        raise ValueError(f"Unknown file: {filepath}")

    if use_resources or ("PYREALM_USE_LOCAL_DATA" in os.environ):
        return str(resources.files("pyrealm_build_data") / filepath)

    return DATASETS.fetch(filepath)
