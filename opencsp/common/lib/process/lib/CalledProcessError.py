import subprocess


class CalledProcessError(subprocess.CalledProcessError):
    def __str__(self):
        ret = super().__str__()
        if self.stderr != None:
            ret += f" Captured stderr:\n\"{self.stderr}\""
        return ret
