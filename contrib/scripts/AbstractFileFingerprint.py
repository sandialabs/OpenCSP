from abc import ABC
import dataclasses
import os


@dataclasses.dataclass()
class AbstractFileFingerprint(ABC):
    relative_path: str
    """ Path to the file, from the root search directory. Usually something like "opencsp/common/lib/tool". """
    name_ext: str
    """ "name.ext" of the file. """

    @property
    def relpath_name_ext(self):
        return os.path.join(self.relative_path, self.name_ext)

    def eq_aff(self, other: "AbstractFileFingerprint"):
        if not isinstance(other, AbstractFileFingerprint):
            return False
        return self.relative_path == other.relative_path and self.name_ext == other.name_ext
