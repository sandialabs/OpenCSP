import json

import opencsp.common.lib.file.AbstractAttributeParser as aap
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.file_tools as ft

_registered_parser_classes: set[aap.AbstractAttributeParser] = set()


class AttributesManager:
    """A class for managing writing and reading attributes .txt files, such as those found next to saved images.

    The purpose of an attributes file is to record the context of a file in a
    human-readable format (we use JSON for this format). For images, this might
    such things as the date the image was created, the operations that have been
    applied, the experiment it was part of, etc.

    There is a global standard set of parsers that instances of this class are
    built with. Other AbstractAttributeReader implementations should register
    themselves with cls.RegisterClass() when their respective files are
    imported. These default parsers can be overridden with specific instances by
    creating a new instance of this manager (for example, a parser can be
    created with customized interpretation arguments).

    Example of writing an attributes file::

        # img = log_scale(equinox_image)
        # img.save("5w1_9am.png")
        exp_parser = ExperimentParser()
        attr = AttributesManager(exp_parser)
        exp_parser.name = "Spring Equinox Test 2023"
        attr.write_attributes("5w1_9am.txt")

    Example of reading an attributes file::

        # img.load("5w1_9am.png")
        attr = AttributesManager()
        attr.load("5w1_9am.txt")
        exp_parser: ExperimentParser = attr.get_parser(ExperimentParser)
        title = exp_parser.title
    """

    def __init__(self, *parsers: aap.AbstractAttributeParser):
        input_parsers = {parser.__class__: parser for parser in parsers}
        self.specific_parsers: dict[type[aap.AbstractAttributeParser], aap.AbstractAttributeParser] = input_parsers
        self.generic_parsers: dict[type[aap.AbstractAttributeParser], aap.AbstractAttributeParser] = {}

        # add parsers whose instance's haven't been passed in yet
        for parser_class in _registered_parser_classes:
            if self.get_parser(parser_class, error_on_not_found=False) == None:
                self.generic_parsers[parser_class] = parser_class()

    @property
    def parsers(self):
        """Returns the list of all parsers for this instance. Parsers that were
        passed in the constructor or in set_parser() are returned first."""
        ret = list(self.specific_parsers.values())
        ret += list(self.generic_parsers.values())
        return ret

    @classmethod
    def _register_parser_class(cls, parser_class: type[aap.AbstractAttributeParser]):
        _registered_parser_classes.add(parser_class)

    def get_parser(
        self, parser_class: type[aap.AbstractAttributeParser], error_on_not_found=True
    ) -> aap.AbstractAttributeParser:
        """Returns the parser instance matching the given parser_class.

        Example::

            attr = am.AttributesManager()
            attr.load(image_name.replace(".png", ".txt"))
            image_attr = attr.get_parser(iap.ImageAttributeParser)
            print(f"Image {image_name} provides context for the experiment '{image_attr.experiment_name}'.")
        """
        # get the specific parser_class instance
        if parser_class in self.specific_parsers:
            return self.specific_parsers[parser_class]
        if parser_class in self.generic_parsers:
            return self.generic_parsers[parser_class]

        return None

    def set_parser(self, parser: aap.AbstractAttributeParser):
        """Sets the given parser as the default parser for the given class."""
        parser_class = parser.__class__
        if parser_class in self.generic_parsers:
            del self.generic_parsers[parser_class]
        self.specific_parsers[parser_class] = parser

    def get_attributes_dict(self, attributes_file_path_name_ext: str = None):
        """Get the attributes for all this instance's parsers that have contents.

        Parameters
        ----------
        attributes_file_path_name_ext : str, optional
            The name of the file to save the attributes to, by default None

        Returns
        -------
        contents: dict[str, Any]
            The attributes contents, where the key represents a parser and the
            value represents that parser's contents.
        """
        contents: dict[str, any] = {}
        for parser in self.parsers:
            if not parser.has_contents():
                continue
            contents = parser.append_contents_for_writing(attributes_file_path_name_ext, contents)
        return contents

    def save(self, attributes_file_path_name_ext: str, overwrite=False):
        """Saves the attributes from this instance's parsers into the given file.

        Parameters
        ----------
        attributes_file_path_name_ext : str
            The path/name.ext of the file to save to.
        overwrite : bool, optional
            True to overwrite the attributes file if it already exists, by default False

        Raises
        ------
        FileExistsError:
            If the attributes file already exists and overwrite is False
        """
        # check the name of the file
        if not attributes_file_path_name_ext.endswith(".txt"):
            lt.debug(
                f"In AttributesManager.save(): it is highly suggested that the attributes file name end with '.txt', but {attributes_file_path_name_ext=}"
            )

        # check if the file already exists
        if ft.file_exists(attributes_file_path_name_ext):
            if not overwrite:
                lt.error_and_raise(
                    FileExistsError,
                    f"Error in AttributesManager.save(): " + f"file {attributes_file_path_name_ext} already exists!",
                )

        # collect the contents to be saved
        contents = self.get_attributes_dict(attributes_file_path_name_ext)
        contents = json.dumps(contents, indent=4)

        # save the attributes file
        path, name, ext = ft.path_components(attributes_file_path_name_ext)
        ft.create_directories_if_necessary(path)
        with open(attributes_file_path_name_ext, "w") as fout:
            fout.write(contents)

    def load(self, attributes_file_path_name_ext: str):
        """Loads the attributes from the given file into the parsers for this
        instance. The attributes can then be retrieved from the desired parser
        with self.get_parser(), or from all parsers with self.parsers.

        Raises
        ------
        FileExistsError:
            The given file does not exist.
        json.decoder.JSONDecodeError:
            The given file is not in JSON format."""
        # get the raw string value of the file
        str_contents = ""
        if not ft.file_exists(attributes_file_path_name_ext):
            errstr = (
                f"Error in AttributesManager.load(): attributes file '{attributes_file_path_name_ext}' does not exist!"
            )
            lt.debug(errstr)
            raise FileNotFoundError(errstr)
        with open(attributes_file_path_name_ext, "r") as fin:
            str_contents = fin.read()

        # parse the file contents
        try:
            json_contents: dict[str, any] = json.loads(str_contents)
        except json.decoder.JSONDecodeError:
            lt.info(f"In AttributesManager.load(): failed to parse attributes file {attributes_file_path_name_ext}")
            raise
        for parser in self.parsers:
            json_contents = parser.parse_attributes_file(attributes_file_path_name_ext, str_contents, json_contents)
