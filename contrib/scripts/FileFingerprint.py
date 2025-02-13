import dataclasses
import hashlib
import os
import sys

import opencsp.common.lib.file.CsvInterface as ci
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt

sys.path.append(os.path.join(orp.opencsp_code_dir(), ".."))
import contrib.scripts.AbstractFileFingerprint as aff  # nopep8


@dataclasses.dataclass()
class FileFingerprint(ci.CsvInterface, aff.AbstractFileFingerprint):
    # relative_path: str
    # name_ext: str
    size: int
    """ Size of the file, in bytes """
    hash_hex: str
    """ The latest hashlib.sha256([file_contents]).hexdigest() of the file. """

    @staticmethod
    def csv_header(delimeter=",") -> str:
        """Static method. Takes at least one parameter 'delimeter' and returns the string that represents the csv header."""
        keys = list(dataclasses.asdict(FileFingerprint("", "", "", "")).keys())
        return delimeter.join(keys)

    def to_csv_line(self, delimeter=",") -> str:
        """Return a string representation of this instance, to be written to a csv file. Does not include a trailing newline."""
        values = list(dataclasses.asdict(self).values())
        return delimeter.join([str(value) for value in values])

    @classmethod
    def from_csv_line(cls, data: list[str]) -> tuple["FileFingerprint", list[str]]:
        """Construct an instance of this class from the pre-split csv line 'data'. Also return any leftover portion of the csv line that wasn't used."""
        root, name_ext, size, hash_hex = data[0], data[1], data[2], data[3]
        size = int(size)
        return cls(root, name_ext, size, hash_hex), data[4:]

    @classmethod
    def for_file(cls, root_path: str, relative_path: str, file_name_ext: str):
        norm_path = ft.norm_path(os.path.join(root_path, relative_path, file_name_ext))
        file_size = ft.file_size(norm_path)
        with open(norm_path, "rb") as fin:
            file_hash = hashlib.sha256(fin.read()).hexdigest()
        return cls(relative_path, file_name_ext, file_size, file_hash)

    def __lt__(self, other: "FileFingerprint"):
        if not isinstance(other, FileFingerprint):
            lt.error_and_raise(TypeError, f"'other' is not of type FileFingerprint but instead of type {type(other)}")
        if self.relative_path == other.relative_path:
            return self.name_ext < other.name_ext
        return self.relative_path < other.relative_path
