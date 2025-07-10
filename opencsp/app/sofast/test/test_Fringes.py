"""Unit test suite to test Fringes class"""

from opencsp.app.sofast.lib.Fringes import Fringes


def test_fringe():
    # Create fringe object
    fringe = Fringes([2.0], [2.0])

    # Create frame
    range_ = [25, 250]
    frame = fringe.get_frames(100, 100, "uint8", range_)

    # Test number of fringes
    assert 8 == frame.shape[2]

    # Test min and max of frame
    assert range_[0] == frame.min()
    assert range_[1] == frame.max()
