import datetime as dt
from typing import Callable, Literal

from numpy.typing import NDArray

import opencsp.app.sofast.lib.DistanceOpticScreen as dos
from opencsp.app.sofast.lib.MeasurementSofastFixed import MeasurementSofastFixed
from opencsp.app.sofast.lib.PatternSofastFixed import PatternSofastFixed
from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.log_tools as lt


class SystemSofastFixed:
    """Class that orchestrates the running of a SofastFixed system"""

    def __init__(self, image_acquisition: ImageAcquisitionAbstract) -> "SystemSofastFixed":
        """
        Instantiates SystemSofastFixed class.

        Parameters
        ----------
        image_acquisition : ImageAcquisition
            Loaded ImageAcquisition object.
        """
        # Import here to avoid circular dependencies
        import opencsp.app.sofast.lib.sofast_common_functions as scf

        # Get defaults
        image_acquisition, image_projection = scf.get_default_or_global_instances(
            image_acquisition_default=image_acquisition
        )
        # Validate input
        if image_projection is None or image_acquisition is None:
            lt.error_and_raise(RuntimeError, "Both ImageAcquisiton and ImageProjection must both be loaded.")
        self.image_acquisition = image_acquisition

        # Show crosshairs
        image_projection.show_crosshairs()

        # Initialize queue
        self._queue_funcs = []

        # Instantiate attributes
        self.image_measurement: NDArray = None
        """Measurement image, 2d numpy array"""
        self.pattern_sofast_fixed: PatternSofastFixed = None
        """The PatternSofastFixed object used to define the fixed pattern image"""
        self.pattern_image: NDArray = None
        """The fixed pattern image being projected. NxMx3 numpy array"""

        self.dtype = "uint8"
        """The data type of the image being projected"""
        self.max_int = 255
        """The value that corresponds to pure white"""
        self.dot_shape: Literal["circle", "square"] = "circle"
        """The shape of the fixed pattern dots"""
        self.image_delay = image_projection.display_data.image_delay_ms  # ms
        """The delay after displaying the image to capturing an image, ms"""

    def set_pattern_parameters(self, width_pattern: int, spacing_pattern: int):
        """Sets the parameters of pattern_sofast_fixed

        Upon completion, runs self.run_next_in_queue()

        Parameters
        ----------
        width_pattern : int
            Width of each Fixed Pattern marker in the image, pixels
        spacing_pattern : int
            Spacing between (not center-to-center) pattern markers, pixels
        """
        lt.debug(f"SystemSofastFixed fixed pattern dot width set to {width_pattern:d} pixels")
        lt.debug(f"SystemSofastFixed fixed pattern dot spacing set to {spacing_pattern:d} pixels")
        image_projection = ImageProjection.instance()
        size_x = image_projection.display_data.active_area_size_x
        size_y = image_projection.display_data.active_area_size_y
        self.pattern_sofast_fixed = PatternSofastFixed(size_x, size_y, width_pattern, spacing_pattern)
        self.pattern_image = self.pattern_sofast_fixed.get_image(self.dtype, self.max_int, self.dot_shape)

        self.run_next_in_queue()

    def project_pattern(self) -> None:
        """Projects the fixed dot pattern

        Upon completion, runs self.run_next_in_queue()
        """
        lt.debug("SystemSofastFixed pattern projected")

        # Check pattern has been defined
        if self.pattern_image is None:
            lt.error_and_raise(
                ValueError, "The SofastFixed pattern has not been set. Run self.set_pattern_parameters() first."
            )
            return

        # Project pattern
        image_projection = ImageProjection.instance()
        image_projection.display_image_in_active_area(self.pattern_image)
        self.run_next_in_queue()

    def capture_image(self) -> None:
        """Captures the measurement image

        Upon completion, runs self.run_next_in_queue()
        """
        lt.debug("SystemSofastFixed capturing camera image")
        self.image_measurement = self.image_acquisition.get_frame()

        self.run_next_in_queue()

    def get_measurement(
        self,
        v_measure_point_facet: Vxyz,
        dist_optic_screen: float,
        origin: Vxy,
        date: dt.datetime = None,
        name: str = "",
    ) -> MeasurementSofastFixed:
        """Returns the SofastFixedMeasurement object"""
        dist_optic_screen_measure = dos.DistanceOpticScreen(v_measure_point_facet, dist_optic_screen)
        return MeasurementSofastFixed(self.image_measurement, dist_optic_screen_measure, origin, date, name)

    def pause(self) -> None:
        """Pause by the given amount defined by self.image_delay (in ms)

        Upon completion, runs self.run_next_in_queue()
        """
        lt.debug(f"SystemSofastFixed pausing by {self.image_delay:.0f} ms")
        image_projection = ImageProjection.instance()
        image_projection.root.after(self.image_delay, self.run_next_in_queue)

    def run_measurement(self) -> None:
        """Runs the measurement sequence
        - projects the fixed pattern image
        - captures an image.

        If the fixed pattern image is already projected, use self.capture_image() instead.
        Upon completion, runs self.run_next_in_queue()
        """
        lt.debug("SystemSofastFixed starting measurement sequence")

        funcs = [self.project_pattern, self.pause, self.capture_image]
        self.prepend_to_queue(funcs)
        self.run_next_in_queue()

    def close_all(self):
        """Closes all windows and cameras"""
        # Close image acquisition
        with et.ignored(Exception):
            self.image_acquisition.close()

        # Close window
        with et.ignored(Exception):
            image_projection = ImageProjection.instance()
            image_projection.close()

    def prepend_to_queue(self, funcs: list[Callable]) -> None:
        """Prepends the current list of functions to the queue"""
        self._queue_funcs = funcs + self._queue_funcs

    def set_queue(self, funcs: list[Callable]) -> None:
        """Sets the queue to given list of Callables"""
        self._queue_funcs = funcs

    def run_next_in_queue(self) -> None:
        """Runs the next funtion in the queue and removes from queue"""
        if len(self._queue_funcs) > 0:
            func = self._queue_funcs.pop(0)

            try:
                func()
            except Exception as error:
                lt.error(repr(error))
                self.run_next_in_queue()

    def run(self) -> None:
        """
        Instantiates the system by starting the mainloop of the ImageProjection object.
        """
        self.run_next_in_queue()
        image_projection = ImageProjection.instance()
        image_projection.root.mainloop()
