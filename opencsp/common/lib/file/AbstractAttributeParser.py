from abc import ABC, abstractmethod
import copy
import json

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.typing_tools as tt


class AbstractAttributeParser(ABC):
    """Class for parsing attributes related to specific data processing steps,
    for writing the context of that data processing to adjacent 'attributes' .txt
    files.

    Inhereting classes need to implement the following methods::

        def attributes_key(self) -> str:
            raise NotImplementedError()

        def set_defaults(self, other: 'InheretingClassName'):
            raise NotImplementedError()

        def has_contents(self) -> bool:
            raise NotImplementedError()

        def parse_my_contents(self, file_path_name_ext: str, raw_contents: str, my_contents: any):
            raise NotImplementedError()

        def my_contents_to_json(self, file_path_name_ext: str) -> any:
            raise NotImplementedError()

    In each inheriting class's file, after the class definition, be sure to also include::

        MyAttributesClassName.RegisterClass()

    See also:
    ---------
    AttributesManager.py
    """

    @abstractmethod
    def attributes_key(self) -> str:
        """Returns the key this class (or this instance) uses to index into the
        attributes file. This key should closely match the class name."""
        pass

    @abstractmethod
    def set_defaults(self, other: "AbstractAttributeParser"):
        """Replaces this instance's None-valued contents with non-None-valued
        contents from the given other of the same parser type."""
        raise NotImplementedError()

    @abstractmethod
    def has_contents(self) -> bool:
        """Returns True if there are contents that have been set for this
        instance. False to skip this instance when saving out an attributes
        file."""
        pass

    @abstractmethod
    def parse_my_contents(self, file_path_name_ext: str, raw_contents: str, my_contents: any):
        """Parse this attribute parser's specific contents.

        Parameters
        ----------
        file_path_name_ext : str
            The path to the attributes file. May be None.
        raw_contents : str
            The unparsed string contents of the attributes file.
        my_contents : any
            The JSON interpretted value for the attributes_key() in the raw_contents."""
        pass

    @abstractmethod
    def my_contents_to_json(self, file_path_name_ext: str) -> any:
        """Prepare the contents from this parser to be written to an attributes file.
        If has_contents() == False, then this parser will be skipped.

        _extended_summary_

        Parameters
        ----------
        file_path_name_ext : str
            The path to the attributes file. May be None.

        Returns
        -------
        my_contents: dict|list|str|float
            The contents to be recorded. Should be json-ifiable.
        """
        pass

    def parse_attributes_file(
        self, file_path_name_ext: str, raw_contents: str, json_contents: dict[str, any]
    ) -> dict[str, any]:
        """Parse this attribute parser's specific contents out of the given
        json_contents. Once this parser's contents have been registered with
        this instance, then they can be removed from the dict and the modified
        dict returned.

        Parameters
        ----------
        file_path_name_ext : str
            The path to the attributes file. May be None.
        raw_contents : str
            The unparsed string contents of the attributes file.
        json_contents : dict[str, any]
            The JSON interpretted representation of the attributes file.

        Returns
        -------
        dict[str, any]
            The modified json_contents, with this parser's contents stripped out.
        """
        if self.attributes_key() not in json_contents:
            return json_contents
        my_contents: dict[str, any] = json_contents[self.attributes_key()]
        json_contents = copy.copy(json_contents)
        del json_contents[self.attributes_key()]

        self.parse_my_contents(file_path_name_ext, raw_contents, my_contents)

        return json_contents

    def save(self, attributes_file_path_name_ext: str, overwrite=None):
        """Loads and parses the given file, then writes the file back with this instance's contents.

        Parameters
        ----------
        attributes_file_path_name_ext : str
            The name of the file to load from + save to.
        overwrite : bool, optional
            Whether to overwrite the existing file, if any. If None, then the
            existing file will be read first, then overwritten. By default None.
        """
        import opencsp.common.lib.file.AttributesManager as am

        old_attr: am.AttributesManager = None
        if ft.file_exists(attributes_file_path_name_ext):
            try:
                # load the file's current contents
                old_attr = am.AttributesManager()
                old_attr.load(attributes_file_path_name_ext)

                # Replace the parser instance for the attribute manager.
                self.set_defaults(old_attr.get_parser(self.__class__))
                old_attr.set_parser(self)

                # Default to overwrite, to update the file's contents with this
                # parser's contents.
                if overwrite == None:
                    overwrite = True
            except json.decoder.JSONDecodeError:
                old_attr = None
        attr = tt.default(old_attr, am.AttributesManager(self))

        # save out the file with the updated contents
        attr.save(attributes_file_path_name_ext, overwrite=overwrite)

    def load(self, attributes_file_path_name_ext: str):
        """Loads and parses the given file, populating this instance with that file's contents.

        Raises
        ------
        json.decode.JSONDecodeError:
            If parsing the given file fails
        """
        import opencsp.common.lib.file.AttributesManager as am

        attr = am.AttributesManager(self)
        attr.load(attributes_file_path_name_ext)

    def append_contents_for_writing(self, file_path_name_ext: str, contents: dict[str, any]) -> dict[str, any]:
        """Add the contents from this parser to be written to an attributes file.
        If has_contents() == False, then this parser will be skipped.

        Parameters
        ----------
        file_path_name_ext : str
            The name of the file to be saved to. May be None.
        contents : dict[str,any]
            The current contents to be written out.

        Returns
        -------
        dict[str,any]
            The modified contents dict with the additional contents for this parser.
        """
        if not self.has_contents():
            return contents
        my_contents = self.my_contents_to_json(file_path_name_ext)
        if my_contents == None:
            lt.debug(
                f"In {self.__class__.__name__}.my_contents_to_json(): "
                + "self.has_contents() returned True, but my_contents_to_json() returned None!"
            )
        ret = copy.copy(contents)
        ret[self.attributes_key()] = my_contents
        return ret

    @classmethod
    def RegisterClass(cls):
        import opencsp.common.lib.file.AttributesManager as am

        am.AttributesManager._register_parser_class(cls)
