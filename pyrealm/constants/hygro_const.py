"""The hygro_const module TODO."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import ConstantsClass


@dataclass(frozen=True)
class HygroConst(ConstantsClass):
    r"""Constants for hygrometric functions.

    This data class provides constants used in :mod:`~pyrealm.hygro`, which includes
    hygrometric conversions. The values for
    :attr:`~pyrealm.param_classes.HygroParams.magnus_coef`` be set directly or by
    selecting one of ``Allen1998``, ``Alduchov1996`` or ``Sonntag1990`` as
    :attr:`~pyrealm.param_classes.HygroParams.magnus_option``. The default setting is to
    use the ``Sonntag1990`` parameters.

    """

    magnus_coef: NDArray[np.float32]
    """Three coefficients of the Magnus equation for saturated vapour pressure."""
    mwr: float = 0.622
    """The ratio molecular weight of water vapour to dry air (:math:`MW_r`, -)"""
    magnus_option: Optional[str] = None
    """Choice of Magnus equation parameterisation."""

    def __post_init__(self) -> None:
        """Populate parameters from init settings.

        This checks the init inputs and populates ``magnus_coef`` from the presets
        if no magnus_coef is specified.

        Returns:
            None
        """
        alts = dict(
            Allen1998=np.array((610.8, 17.27, 237.3)),
            Alduchov1996=np.array((610.94, 17.625, 243.04)),
            Sonntag1990=np.array((611.2, 17.62, 243.12)),
        )

        # Note that object is being used here to update a frozen dataclass

        # Set default to Sonntag1990
        if not hasattr(self, "magnus_coef") and self.magnus_option is None:
            object.__setattr__(self, "magnus_coef", alts["Sonntag1990"])
            return

        # Parse other options
        if self.magnus_option is not None:
            if self.magnus_option not in alts:
                raise (ValueError(f"magnus_option must be one of {list(alts.keys())}"))

            object.__setattr__(self, "magnus_coef", alts[self.magnus_option])
            return

        if self.magnus_coef is not None and len(self.magnus_coef) != 3:
            raise TypeError("magnus_coef must be a tuple of 3 numbers")
