class ReferenceFrame():
    """
    Tracks the displacement and rotation of different reference frames
    """
    def __init__(self, dx: float, dy: float, dz: float, rx: float, ry: float, rz: float) -> None:
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.rx = rx
        self.ry = ry
        self.rz = rz