"""The flora module implements:

* The Flora class. This is pydantic data model that provides a set of plant functional
  types (PFT), defined as a set of functional traits for each PFT. The class defines a
  specific set of traits used for demographic modelling in pyrealm, mostly the
  parameterisation of the T Model, but also some additional parameters for modelling
  crown shape.

  Instances are created by providing a dictionary of lists of trait values - the custom
  model validation checks that the lists are of equal length. Any missing fields are
  filled in using the default values, unless the model context specifies 'strict'
  validation when the user must provide all fields.

  The class provides `from_csv` to generate an instance from trait data stored in a CSV
  file.

* Two functions to calculate computed traits (``q_m`` and ``z_max_prop``).
"""  # noqa: D415


# TODO

from __future__ import annotations

from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ValidationError,
    ValidationInfo,
    computed_field,
    model_validator,
)


def calculate_crown_q_m(
    m: NDArray[np.floating], n: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Calculate the crown scaling trait ``q_m``.

    The value of q_m is a constant crown scaling parameter derived from the ``m`` and
    ``n`` attributes defined for a plant functional type.

    Args:
        m: Crown shape parameter
        n: Crown shape parameter
    """
    return (
        m
        * n
        * ((n - 1) / (m * n - 1)) ** (1 - 1 / n)
        * (((m - 1) * n) / (m * n - 1)) ** (m - 1)
    )


def calculate_crown_z_max_proportion(
    m: NDArray[np.floating], n: NDArray[np.floating]
) -> NDArray[np.floating]:
    r"""Calculate the z_m trait.

    The z_m proportion (:math:`p_{zm}`) is the constant proportion of stem height at
    which the maximum crown radius is found for a given plant functional type.

    .. math::

        p_{zm} = \left(\dfrac{n-1}{m n -1}\right)^ {\tfrac{1}{n}}

    Args:
        m: Crown shape parameter
        n: Crown shape parameter
    """

    return ((n - 1) / (m * n - 1)) ** (1 / n)


class Flora(pd.DataFrame):
    """The Cohorts class.

    The Flora class is simply an alias for a {class}`pandas.DataFrame`.
    """


class FloraValidator(BaseModel):
    """The Flora class.

    This dataclass implements the set of traits required to define a plant functional
    type for use in ``pyrealm``.

    * Most traits are taken from the definition of the T Model of plant growth and GPP
      allocation :cite:`Li:2014bc`.
    * The foliage maintenance respiration fraction was not explicitly included in
      :cite:t:`Li:2014bc` - there was assumed to be a 10% penalty on GPP before
      calculating the other component - but has been explicitly included here.
    * This implementation adds two further crown shape parameters (``m`` and ``n`` and
      ``f_g``). The first two are then used to calculate two constant derived attributes
      (``q_m`` and ``z_max_ratio``) that define the vertical distribution of the crown.
      The last parameter (``f_g``) is the crown gap fraction, that defines the vertical
      distribution of leaves within the crown. This crown model parameterisation
      follows the implementation developed in the PlantFATE model :cite:`joshi:2022a`.
    * The branch turnover rate (``tau_b``) has been added to capture branch fall and
      loss of other woody tissue forming part of normal tree growth. This defaults to
      infinity to match the expectations of the original T Model in which woody tissue
      is never lost to turnover. There is no specific branch mass in the T model, so
      this is the rate at which total stem biomass turns over through branch loss.
    """

    pft_name: tuple[str, ...] = ("default",)
    r"""The name of the plant functional type."""
    a_hd: tuple[float, ...] = (116.0,)
    r"""Initial slope of height-diameter relationship (:math:`a`, -)"""
    ca_ratio: tuple[float, ...] = (390.43,)
    r"""Initial ratio of crown area to stem cross-sectional area (:math:`c`, -)"""
    h_max: tuple[float, ...] = (25.33,)
    r"""Maximum tree height (:math:`H_m`, m)"""
    rho_s: tuple[float, ...] = (200.0,)
    r"""Sapwood density (:math:`\rho_s`, kg Cm-3)"""
    lai: tuple[float, ...] = (1.8,)
    """Leaf area index within the crown (:math:`L`,  -)"""
    sla: tuple[float, ...] = (14.0,)
    r"""Specific leaf area (:math:`\sigma`,  m2 kg-1 C)"""
    tau_f: tuple[float, ...] = (4.0,)
    r"""Foliage turnover time (:math:`\tau_f`,years)"""
    tau_r: tuple[float, ...] = (1.04,)
    r"""Fine-root turnover time (:math:`\tau_r`,  years)"""
    tau_b: tuple[float, ...] = (np.inf,)
    r"""Branch turnover time (:math:`\tau_b`,  years)"""
    par_ext: tuple[float, ...] = (0.5,)
    r"""Extinction coefficient of photosynthetically active radiation (PAR) (:math:`k`,
     -)"""
    yld: tuple[float, ...] = (0.6,)
    r"""Yield factor (:math:`y`,  -)"""
    zeta: tuple[float, ...] = (0.17,)
    r"""Ratio of fine-root mass to foliage area (:math:`\zeta`, kg C m-2)"""
    resp_r: tuple[float, ...] = (0.913,)
    r"""Fine-root specific respiration rate (:math:`r_r`, year-1)"""
    resp_s: tuple[float, ...] = (0.044,)
    r"""Sapwood-specific respiration rate (:math:`r_s`,  year-1)"""
    resp_f: tuple[float, ...] = (0.1,)
    r"""Foliage maintenance respiration fraction (:math:`r_f`,  -)"""
    m: tuple[float, ...] = (2,)
    r"""Crown shape parameter (:math:`m`, -)"""
    n: tuple[float, ...] = (5,)
    r"""Crown shape parameter (:math:`n`, -)"""
    f_g: tuple[float, ...] = (0.05,)
    r"""Crown gap fraction (:math:`f_g`, -)"""

    # This decorator order for computed fields is recommended by pydantic but mypy
    # objects, so mute the warnings.

    @computed_field  # type: ignore[prop-decorator]
    @property
    def q_m(self) -> tuple[float, ...]:
        """Scaling factor to derive maximum crown radius from crown area."""

        # A bit odd here - the validator uses tuples of values, but the functions use
        # np.arrays. It makes more sense to keep those standalone functions as np inputs
        # so here we shimmy to np and back. The alternative is to use numpy-pydantic.
        return tuple(
            calculate_crown_q_m(m=np.array(self.m), n=np.array(self.n)).tolist()
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def z_max_prop(self) -> tuple[float, ...]:
        """Proportion of stem height at which maximum crown radius is found."""

        # See comment on q_m property.
        return tuple(
            calculate_crown_z_max_proportion(
                m=np.array(self.m), n=np.array(self.n)
            ).tolist()
        )

    @model_validator(mode="after")
    def model_validation(self, info: ValidationInfo) -> Self:
        """Implements strict validation.

        This validator uses the validation context to toggle a strict validation mode
        where all of the input fields need to be specified and cannot be filled from
        default.
        """

        # Detect missing fields
        fields_with_defaults = {
            k for k, v in type(self).model_fields.items() if not v.is_required()
        }
        missing = fields_with_defaults - self.model_fields_set

        # Implement strict mode
        if missing and isinstance(info.context, dict) and info.context.get("strict"):
            raise ValueError(f"Missing traits in strict mode: {', '.join(missing)}")

        # Check field lengths of provided data
        field_lengths = set([len(getattr(self, nm)) for nm in self.model_fields_set])
        if len(field_lengths) > 1:
            raise ValueError(
                f"Unequal field lengths: {', '.join([str(it) for it in field_lengths])}"
            )

        # If at least some fields provided and the length is not one, then adjust the
        # lengths of defaults being used for the missing fields
        length = field_lengths.pop() if len(field_lengths) else 1
        if missing and length > 1:
            for field in missing:
                setattr(self, field, getattr(self, field) * length)

        return self


def load_flora_from_csv(
    path: Path, strict: bool = False, validator: type[FloraValidator] = FloraValidator
) -> Flora:
    """Create a Flora object from a CSV file.

    Args:
        path: A path to a CSV file of plant functional type definitions.
        strict: Require that all traits are specified in the input file.
        validator: A pydantic class used to validate the data.
    """

    try:
        data = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.ParserError) as excep:
        raise excep

    return create_flora(data.to_dict(orient="list"), strict=strict, validator=validator)


def create_flora(
    data: dict,
    strict: bool = False,
    validator: type[FloraValidator] = FloraValidator,
) -> Flora:
    """Create a Flora object from a dictionary of data.

    Args:
        data: A dictionary providing plant functional trait data.
        strict: Require that all traits are specified in the input data.
        validator: A pydantic class used to validate the data.
    """
    try:
        validated_data = validator.model_validate(data, context={"strict": strict})
    except ValidationError as excep:
        raise excep

    return Flora(validated_data.model_dump())
