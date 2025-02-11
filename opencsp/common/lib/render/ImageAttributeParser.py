import datetime
import os

import opencsp.common.lib.file.AbstractAttributeParser as aap
import opencsp.common.lib.file.AttributesManager as am
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.typing_tools as tt


class ImageAttributeParser(aap.AbstractAttributeParser):
    """
    Subclass of AbstractAttributeParser that adds the following extra attributes to the attributes file:

        - current_image_source (str): The most recent filename (or network streamed name) that this image was loaded
          from.
        - original_image_source (str): The definitive filename (or network streamed name) that this image was loaded
          from. This is usually going to be the name of the original file, such as "Nikon_2024-04-16.png".
        - date_collected (datetime): Notes from the specific image processors about this image.
        - experiment_name (str): The name of the measurement or experiment that this image was collected as a part of.
        - notes (str): Extra notes about the image, typically added by the user.
    """

    def __init__(
        self,
        current_image_source: str = None,
        original_image_source: str = None,
        date_collected: datetime.datetime = None,
        experiment_name: str = None,
        notes: str = None,
    ):
        """Manager for the most common attributes associated with processed
        images, for maintaining the context and history of the image.

        Parameters
        ----------
        current_image_source : str, optional
            The current source of the image, probably a file name. If not None,
            then this will attempt to populate the rest of the fields by looking
            for a sibling ".txt" attributes file. By default None.
        original_image_source : str, optional
            The original source of the image, probably a file name. If not None,
            and current_image_source is None, then this will attempt to populate
            the rest of the fields by looking for a sibling ".txt" attributes
            file. By default None.
        date_collected : datetime, optional
            The date (and time) that the original image was collected at, by default None
        experiment_name : str, optional
            A descriptive name of the experiment that the original image was
            collected in, if it was part of an experiment. By default None.
        notes : str, optional
            Any additional notes that would help with understanding the context
            or history of the image, by default None
        """
        self.current_image_source = current_image_source
        self.original_image_source = original_image_source
        self.date_collected = date_collected
        self.experiment_name = experiment_name
        self.notes = notes

        # retrieve any available attributes from the associated attributes file
        self._previous_attr: am.AttributesManager = None
        with et.ignored(Exception):
            if self.current_image_source is not None:
                opath, oname, oext = ft.path_components(self.current_image_source)
            else:
                opath, oname, oext = ft.path_components(self.original_image_source)

            attributes_file = os.path.join(opath, f"{oname}.txt")
            with et.ignored(Exception):
                self._previous_attr = am.AttributesManager()
                self._previous_attr.load(attributes_file)
        if self._previous_attr != None:
            prev_image_attr: ImageAttributeParser = self._previous_attr.get_parser(self.__class__)

            # Sanity check: are we trying to overwrite the "original_image_source" value?
            if prev_image_attr != None:
                if prev_image_attr.original_image_source != None:
                    if self.original_image_source != None:
                        lt.error_and_raise(
                            ValueError,
                            "Error in ImageAttributeParser.__init__(): "
                            + f"can't overwrite existing original_image_source value ('{prev_image_attr.original_image_source}') "
                            + f"with new original_image_source ('{self.original_image_source}')!",
                        )

            self.set_defaults(prev_image_attr)
        pass

    def attributes_key(self) -> str:
        return "image attributes"

    def set_defaults(self, other: 'ImageAttributeParser'):
        # Specifically for image attributes, for original image source, we
        # really want to maintain the absolute original image source throughout
        # all processing steps and files.
        self.original_image_source = tt.default(other.original_image_source, self.original_image_source)
        # the rest of these attributes can be set as normal
        self.current_image_source = tt.default(self.current_image_source, other.current_image_source)
        self.date_collected = tt.default(self.date_collected, other.date_collected)
        self.experiment_name = tt.default(self.experiment_name, other.experiment_name)
        self.notes = tt.default(self.notes, other.notes)

    def has_contents(self) -> bool:
        return (
            (self.current_image_source != None)
            or (self.original_image_source != None)
            or (self.date_collected != None)
            or (self.experiment_name != None)
            or (self.notes != None)
        )

    def parse_my_contents(self, file_path_name_ext: str, raw_contents: str, my_contents: any):
        self.current_image_source = my_contents['current_image_source']
        self.original_image_source = my_contents['original_image_source']
        self.date_collected = None
        if my_contents['date_collected'] != None:
            self.date_collected = datetime.datetime.fromisoformat(my_contents['date_collected'])
        self.experiment_name = my_contents['experiment_name']
        self.notes = my_contents['notes']

    def my_contents_to_json(self, file_path_name_ext: str) -> any:
        date_collected_str = tt.default(lambda: self.date_collected.isoformat(), None)
        ret = {
            'current_image_source': self.current_image_source,
            'original_image_source': self.original_image_source,
            'date_collected': date_collected_str,
            'experiment_name': self.experiment_name,
            'notes': self.notes,
        }
        return ret


ImageAttributeParser.RegisterClass()
