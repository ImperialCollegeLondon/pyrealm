"""Many of the functions and classes in `pyrealm` have an underlying set of
parameters that will typically be held constant. This includes true universal constants,
such as the molar mass of Carbon, but also a large number of calculated parameters from
the literature.

These underlying parameters are not hardcoded, but rather dataclasses for constants  are
defined for different models within `pyrealm`. The defaults in these dataclasses can be
altered and can also be loaded and saved to configuration files. The
:class:`~pyrealm.constants.base.ConstantsClass` base class provides the basic load and
save methods and then individual subclasses define the constants and default values for
each class.

This implementation has the following desired features:

1. Ability to use a obj.attr notation rather than obj['attr'].
2. Ability to freeze values to avoid them being edited in use - distinctly paranoid!
3. Ability to set a default mapping with default values.
4. Typing to set expected types on default values.
5. Simple export/import methods to go to from dict / JSON.
6. Is a class, to allow __repr__ and other methods.
"""  # noqa D210, D415

import json
from dataclasses import asdict, dataclass

from dacite import from_dict


@dataclass(frozen=True)
class ConstantsClass:
    """Base class for model constants.

    This base class provides a consistent interface for creating and exporting data from
    model constant classes. It defines methods to create instances from a dictionary of
    values and from a JSON file of values. It also defines methods to export an existing
    instance of a constants class to a dictionary or a JSON file.

    All model constant classes use the :func:`~dataclasses.dataclass` decorator as the
    basic structure. All fields in these classes are defined with default values, so
    instances can be defined using partial dictionaries of alternative values.
    """

    def to_dict(self) -> dict:
        """Return a parameter dictionary.

        This method returns the attributes and values used in an instance of a parameter
        class object as a dictionary.

        Returns:
            A dictionary
        """
        return asdict(self)

    def to_json(self, filename: str) -> None:
        """Export a parameter set to JSON.

        This method writes the attributes and values used in an instance of a
        parameter class object to a JSON file.

        Args:
            filename: A path to the JSON file to be created.

        Returns:
            None
        """
        with open(filename, "w") as outfile:
            json.dump(self.to_dict(), outfile, indent=4)

    @classmethod
    def from_dict(cls, data: dict) -> "ConstantsClass":
        """Create a ConstantsClass subclass instance from a dictionary.

        Generates a :class:`~pyrealm.constants.base.ConstantsClass` subclass instance
        using the data provided in a dictionary to override default values

        Args:
          data: A dictionary, keyed by parameter class attribute names, providing values
            to use in a new instance.

        Returns:
            An instance of the subclass.
        """
        return from_dict(cls, data)

    @classmethod
    def from_json(cls, filename: str) -> "ConstantsClass":
        """Create a ParamClass instance from a JSON file.

        Generates a :class:`~pyrealm.constants.base.ConstantsClass` subclass instance
        using the data provided in a  JSON file.

        Args:
          filename: The path to a JSON formatted file containing a set of values keyed
            by parameter class attribute names to use in a new instance.

        Returns:
            An instance of the parameter class.
        """
        with open(filename) as infile:
            json_data = json.load(infile)

        return cls.from_dict(json_data)
