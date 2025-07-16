""" """

from typing import Iterable


class RenderControlEnsemble:
    """
    Render control for collections of named objects.

    Provides a default render copntrol, with exceptions for objets with specific names.

    Multiple classes of exceptions can be defineid, each with its own specialized render style.

    Render styles may be of arbitrary type:  RenderControlFacet, RenderControlHeliostat, etc.

    """

    def __init__(self, default_style):

        super(RenderControlEnsemble, self).__init__()

        self.default_style = default_style
        self.special_style_entries = []

    def add_special_name(self, input_name: str | float, style: any):
        # Input names may be strings or numbers.
        # This can sometimes lead to confusion where a numerical name doesn't match its string equivalent.
        # Ensure that all names are stored as strings.
        name = str(input_name)

        # Check input.
        for entry in self.special_style_entries:
            if name in entry[0]:
                print(
                    'In RenderControlEnsemble.add_special_name(), name="'
                    + str(name)
                    + '" already has a special style defined.'
                )
                assert False

        # Add special entry.
        entry = [[name], style]
        self.special_style_entries.append(entry)

    def add_special_names(self, input_names: Iterable[str | float], style: any):
        # Input names may be strings or numbers.
        # This can sometimes lead to confusion where a numerical name doesn't match its string equivalent.
        # Ensure that all names are stored as strings.
        names = []
        for input_name in input_names:
            name = str(input_name)
            names.append(name)

        # Check input.
        for entry in self.special_style_entries:
            for name in names:
                if name in entry[0]:
                    print(
                        'In RenderControlEnsemble.add_special_names(), name="'
                        + str(name)
                        + '" already has a special style defined.'
                    )
                    assert False

        # Add special entry.
        entry = [names, style]
        self.special_style_entries.append(entry)

    def style(self, input_name: str | float):
        # Input names may be strings or numbers.
        # This can sometimes lead to confusion where a numerical name doesn't match its string equivalent.
        # Ensure that all names are stored as strings.
        name = str(input_name)

        # See if there is a special style for this name.
        matching_entry = None
        for entry in self.special_style_entries:
            if name in entry[0]:
                matching_entry = entry
                break
        # Return.
        if matching_entry:
            return entry[1]
        else:
            return self.default_style
