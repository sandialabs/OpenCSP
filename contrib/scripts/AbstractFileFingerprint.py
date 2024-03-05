from abc import ABC
import dataclasses


@dataclasses.dataclass()
class AbstractFileFingerprint(ABC):
    relative_path: str
    """ Path to the file, from the root search directory. Usually something like "opencsp/common/lib/tool". """
    name_ext: str
    """ "name.ext" of the file. """

    def eq_aff(self, other: 'AbstractFileFingerprint'):
        if not isinstance(other, AbstractFileFingerprint):
            return False
        return self.relative_path == other.relative_path and \
            self.name_ext == other.name_ext
