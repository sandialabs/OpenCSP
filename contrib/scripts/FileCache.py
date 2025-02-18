import dataclasses
import datetime
import os
import sys

import opencsp.common.lib.file.CsvInterface as ci
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.time_date_tools as tdt
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt

sys.path.append(os.path.join(orp.opencsp_code_dir(), ".."))
import contrib.scripts.AbstractFileFingerprint as aff  # nopep8


@dataclasses.dataclass()
class FileCache(ci.CsvInterface, aff.AbstractFileFingerprint):
    # relative_path: str
    # name_ext: str
    last_modified: str
    """ The system time that the file was last modified at. """

    @staticmethod
    def csv_header(delimeter=",") -> str:
        """Static method. Takes at least one parameter 'delimeter' and returns the string that represents the csv header."""
        keys = list(dataclasses.asdict(FileCache("", "", "")).keys())
        return delimeter.join(keys)

    def to_csv_line(self, delimeter=",") -> str:
        """Return a string representation of this instance, to be written to a csv file. Does not include a trailing newline."""
        values = list(dataclasses.asdict(self).values())
        return delimeter.join([str(value) for value in values])

    @classmethod
    def from_csv_line(cls, data: list[str]) -> tuple["FileCache", list[str]]:
        """Construct an instance of this class from the pre-split csv line 'data'. Also return any leftover portion of the csv line that wasn't used."""
        root, name_ext, last_modified = data[0], data[1], data[2]
        return cls(root, name_ext, last_modified), data[3:]

    @classmethod
    def for_file(cls, root_path: str, relative_path: str, file_name_ext: str):
        norm_path = ft.norm_path(os.path.join(root_path, relative_path, file_name_ext))
        modified_time = datetime.datetime.fromtimestamp(os.stat(norm_path).st_mtime)
        last_modified = modified_time.strftime("%Y-%m-%d %H:%M:%S")
        return cls(relative_path, file_name_ext, last_modified)

    def __hash__(self):
        return hash(self.relative_path)
