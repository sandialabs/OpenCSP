from concurrent.futures import ThreadPoolExecutor
import dataclasses
from typing import Callable

import numpy as np
import numpy.typing as npt

from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.DisplayShape import DisplayShape
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.app.sofast.lib.SystemSofastFringe import SystemSofastFringe
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.deflectometry.Surface2DAbstract import Surface2DAbstract
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
import opencsp.common.lib.geometry.Vxyz as vxyz
import opencsp.common.lib.process.ControlledContext as cc
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.time_date_tools as tdt


@dataclasses.dataclass
class FixedResults:
    pass  # TODO


@dataclasses.dataclass
class FringeResults:
    measurement: MeasurementSofastFringe
    """The collected measurement data"""
    sofast: ProcessSofastFringe
    """The object create for processing the measurement"""
    facet: Facet
    """The facet representation"""
    res: float
    """The resolution of the slopes matrix, in meters"""
    focal_length_x: float
    """Focal length in the x axis"""
    focal_length_y: float
    """Focal length in the y axis"""
    slopes: npt.NDArray[np.float_]
    """X and Y slopes of the facet, as measured. Shape (2, facet_width/res, facet_height/res)"""
    slopes_error: npt.NDArray[np.float_] | None
    """X and Y slopes error of the facet relative to the reference facet. Shape (2, facet_width/res, facet_height/res)"""


class Executor:
    """Class to handle the collection and processing of Sofast measurements."""

    def __init__(self, asynchronous_processing=True):
        """
        Class to handle collection and processing of sofast measurements.

        Collection is handled in the main thread (aka the tkinter thread). Processing is handled either in a separate
        processing thread, or in the calling thread.

        Parameters
        ----------
        asynchronous : bool, optional
            If True then processing is done in a separate thread, if False then it is done on the calling thread. By
            default True.
        """
        self.on_fringe_collected: Callable
        self.on_fringe_processed: Callable[[FringeResults | None, Exception | None], None]
        self.on_fixed_collected: Callable
        self.on_fixed_processed: Callable[[FixedResults | None, Exception | None], None]

        # processing thread
        self.asynchronous_processing = asynchronous_processing
        self._processing_pool = ThreadPoolExecutor(max_workers=1)

        # don't try to close multiple times
        self.is_closed = False

    def __del__(self):
        with et.ignored(Exception):
            self.close()

    def start_collect_fringe(self, controlled_system: cc.ControlledContext[SystemSofastFringe]):
        """Starts collection of fringe measurement image data.

        Once this method is called it returns immediately without waiting for collection and processing to finish. The
        collection is queued up for the main thread (aka the tkinter thread), and on_fringe_collected is called when
        finished.
        """
        # Start the collection
        lt.debug("Executor: collecting fringes")

        # Run the measurement in the main thread (aka the tkinter thread)
        with controlled_system as system:
            system.run_measurement(self.on_fringe_collected)

    def start_process_fringe(
        self,
        controlled_system: cc.ControlledContext[SystemSofastFringe],
        mirror_measure_point: vxyz.Vxyz,
        mirror_measure_distance: float,
        orientation: SpatialOrientation,
        camera: Camera,
        display: DisplayShape,
        facet_data: DefinitionFacet,
        surface: Surface2DAbstract,
        measurement_name: str = None,
        reference_facet: Facet = None,
    ):
        """
        Processes the given fringe collected from system.run_measurement().

        This method is evaluated in this instance's _processing_pool thread, and on_fringe_processed is called from that
        thread once processing has finished.
        """
        lt.debug("Executor: processing fringes")

        name = "fringe_measurement_" + tdt.current_date_time_string_forfile()
        if measurement_name != None:
            name = measurement_name

        def _process_fringes():
            try:
                with controlled_system as system:
                    # Get the measurement
                    measurements = system.get_measurements(mirror_measure_point, mirror_measure_distance, name)
                    measurement = measurements[0]

                    # Apply calibration to the fringe images
                    measurement.calibrate_fringe_images(system.calibration)

                # Process the measurement
                sofast = ProcessSofastFringe(measurements[0], orientation, camera, display)
                # sofast.params.geometry_data_debug.debug_active = True
                sofast.process_optic_singlefacet(facet_data, surface)
                facet: Facet = sofast.get_optic()

                # Get the focal lengths (if a parabolic mirror)
                facet_idx = 0
                surf_coefs = sofast.data_characterization_facet[facet_idx].surf_coefs_facet
                if surf_coefs.size >= 6:
                    if not isinstance(surface, Surface2DParabolic):
                        lt.warn(
                            "Warning in Executor.start_process_fringe(): "
                            + "did not expect a non-parabolic mirror to have a focal point"
                        )
                    focal_length_x, focal_length_y = 1 / 4 / surf_coefs[2], 1 / 4 / surf_coefs[5]
                else:
                    if isinstance(surface, Surface2DParabolic):
                        lt.warn(
                            "Warning in Executor.start_process_fringe(): "
                            + "expected a parabolic mirror to have a focal point"
                        )
                    focal_length_x, focal_length_y = None, None

                # Create interpolation axes
                res = 0.1  # meters
                left, right, bottom, top = facet.axis_aligned_bounding_box
                x_vec = np.arange(left, right, res)  # meters
                y_vec = np.arange(bottom, top, res)  # meters

                # Calculate current mirror slope
                slopes_cur = facet.orthorectified_slope_array(x_vec, y_vec)  # radians

                # Calculate slope difference (error)
                slopes_diff: npt.NDArray[np.float_] = None
                if reference_facet is not None:
                    slopes_ref = reference_facet.orthorectified_slope_array(x_vec, y_vec)  # radians
                    slopes_diff = slopes_cur - slopes_ref  # radians

                # Call the callback
                ret = FringeResults(
                    measurement, sofast, facet, res, focal_length_x, focal_length_y, slopes_cur, slopes_diff
                )
                self.on_fringe_processed(ret, None)
            except Exception as ex:
                lt.error("Error in Executor.start_process_fringe(): " + repr(ex))
                self.on_fringe_processed(None, ex)

        if self.asynchronous_processing:
            self._processing_pool.submit(_process_fringes)
        else:
            _process_fringes()

    def close(self):
        """Closes the processing thread (may take several seconds)"""
        # don't try to close multiple times
        if self.is_closed:
            return
        self.is_closed = True

        with et.ignored(Exception):
            lt.debug("Executor.close(): Stopping processing thread")
            self._processing_pool.shutdown(wait=True, cancel_futures=True)
            self._processing_pool = None
