import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, TypeVar

from opencsp.app.sofast.lib.MeasurementSofastFixed import MeasurementSofastFixed
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.SystemSofastFringe import SystemSofastFringe
from opencsp.app.sofast.lib.SystemSofastFixed import SystemSofastFixed
from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
import opencsp.common.lib.geometry.Vxyz as vxyz
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.time_date_tools as tdt


T = TypeVar('T')


class ControlledContext(Generic[T]):
    def __init__(self, o: T):
        self.o = o
        self.mutex = asyncio.Lock()

    def __enter__(self):
        self.mutex.acquire()
        return self.o

    def __exit__(self, exc_type, exc_value, traceback):
        self.mutex.release()
        return False


class ServerState:
    _instance: ControlledContext['ServerState'] = None

    def __init__(self):
        # systems
        self._system_fixed: ControlledContext[SystemSofastFixed] = None
        self._system_fringe: ControlledContext[SystemSofastFringe] = None

        # measurements
        self._last_measurement_fixed: MeasurementSofastFixed = None
        self._last_measurement_fringe: list[MeasurementSofastFringe] = None
        self.fixed_measurement_name: str = None
        self.fringe_measurement_name: str = None

        # configurations
        self._mirror_measure_point: vxyz.Vxyz = None
        self._mirror_measure_distance: float = None
        self._fixed_pattern_diameter: int = None
        self._fixed_pattern_spacing: int = None

        # statuses
        self._running_measurement_fixed = False
        self._running_measurement_fringe = False
        self._processing_measurement_fixed = False
        self._processing_measurement_fringe = False

        # processing thread
        self._processing_pool = ThreadPoolExecutor(max_workers=1)

        if ServerState._instance is None:
            ServerState._instance = ControlledContext(self)
        else:
            lt.error_and_raise(RuntimeError, "Error in ServerState(): " +
                               "this class is supposed to be a singleton, but another instance already exists!")

    @staticmethod
    def instance() -> ControlledContext['ServerState']:
        return ServerState._instance

    @property
    def system_fixed(self) -> ControlledContext[SystemSofastFixed]:
        if self._system_fixed is None:
            display_data = ImageProjection.instance().display_data
            size_x, size_y = display_data['size_x'], display_data['size_y']
            width_pattern = self._fixed_pattern_diameter
            spacing_pattern = self._fixed_pattern_spacing
            self._system_fixed = ControlledContext(SystemSofastFixed(size_x, size_y, width_pattern, spacing_pattern))
        return self._system_fixed

    @property
    def system_fringe(self) -> ControlledContext[SystemSofastFringe]:
        if self._system_fringe is None:
            self._system_fringe = ControlledContext(SystemSofastFringe())
        return self._system_fringe

    @property
    def projector_available(self) -> bool:
        """
        Returns False for the collection period of a sofast measurement. Returns True when available to start a new
        measurement.
        """
        if self._running_measurement_fringe:
            return False
        elif self._running_measurement_fixed:
            return False
        else:
            return True

    @property
    def has_fixed_measurement(self) -> bool:
        """
        Returns False for the processing or collection periods of a fixed measurement. Returns True when results are
        available.
        """
        if self._last_measurement_fixed is None:
            return False
        elif self._running_measurement_fixed:
            lt.error("Programmer error in ServerState.fixed_measurement_ready(): " +
                     f"expected 'last_measurement' to be None while 'running', but {self._last_measurement_fixed=} {self._running_measurement_fixed}")
            return False
        elif self._processing_measurement_fixed:
            lt.error("Programmer error in ServerState.fixed_measurement_ready(): " +
                     f"expected 'last_measurement' to be None while 'processing', but {self._last_measurement_fixed=} {self._processing_measurement_fixed}")
            return False
        else:
            return True

    @property
    def last_measurement_fixed(self) -> list[MeasurementSofastFixed]:
        if not self.has_fixed_measurement:
            return None
        return self._last_measurement_fixed

    @property
    def has_fringe_measurement(self) -> bool:
        """
        Returns False for the processing or collection periods of a fringe measurement. Returns True when results are
        available.
        """
        if self._last_measurement_fringe is None:
            return False
        elif self._running_measurement_fringe:
            lt.error("Programmer error in ServerState.fringe_measurement_ready(): " +
                     f"expected 'last_measurement' to be None while 'running', but {self._last_measurement_fringe=} {self._running_measurement_fringe}")
            return False
        elif self._processing_measurement_fringe:
            lt.error("Programmer error in ServerState.fringe_measurement_ready(): " +
                     f"expected 'last_measurement' to be None while 'processing', but {self._last_measurement_fringe=} {self._processing_measurement_fringe}")
            return False
        else:
            return True

    @property
    def last_measurement_fringe(self) -> list[MeasurementSofastFringe]:
        if not self.has_fringe_measurement:
            return None
        return self._last_measurement_fringe

    @property
    def busy(self) -> bool:
        """
        Returns True if running a measurement collection or processing a measurement. Returns False when all resources
        all available for starting a new measurement.
        """
        if self._running_measurement_fixed:
            return False
        elif self._running_measurement_fringe:
            return False
        elif self._processing_measurement_fixed:
            return False
        elif self._processing_measurement_fringe:
            return False
        else:
            return True

    def _connect_default_cameras(self):
        pass

    def _load_default_projector(self):
        pass

    def _load_default_calibration(self):
        pass

    def _get_default_mirror_distance_measurement(self):
        pass

    def init_io(self):
        self._connect_default_cameras()
        self._load_default_projector()
        self._load_default_calibration()

    def start_measure_fringes(self, name: str = None) -> bool:
        """Starts collection and processing of fringe measurement image data.

        Returns
        -------
        success: bool
            True if the measurement was able to be started. False if the system resources are busy.
        """
        # Check that system resources are available
        if self._running_measurement_fringe or self._processing_measurement_fringe:
            lt.warn("Warning in server_api.run_measurment_fringe(): " +
                    "Attempting to start another fringe measurement before the last fringe measurement has finished.")
            return False
        if not self.projector_available:
            lt.warn("Warning in server_api.run_measurment_fringe(): " +
                    "Attempting to start a fringe measurement while the projector is already in use.")
            return False

        # Latch the name value
        self.fringe_measurement_name = name

        # Update statuses
        self._last_measurement_fringe = None
        self._running_measurement_fringe = True

        # Start the measurement
        lt.debug("ServerState: collecting fringes")
        with self.system_fringe as sys:
            sys.run_measurement(self._on_collect_fringes_done)

        return True

    def _on_collect_fringes_done(self):
        lt.debug("ServerState: finished collecting fringes")
        if not self._running_measurement_fringe:
            lt.error("Programmer error in server_api._on_collect_fringes_done(): " +
                     "Did not expect for this method to be called while self._running_measurement_fringe was not True!")

        # Update statuses
        self._processing_measurement_fringe = True
        self._running_measurement_fringe = False

        # Start the processing
        self._processing_pool.submit(self._process_fringes)

    def _process_fringes(self):
        lt.debug("ServerState: processing fringes")
        if not self._processing_measurement_fringe:
            lt.error("Programmer error in server_api._process_fringes(): " +
                     "Did not expect for this method to be called while self._processing_measurement_fringe was not True!")

        # process the fringes
        with self.system_fringe as sys:
            name = "fringe_measurement_"+tdt.current_date_time_string_forfile()
            if self.fringe_measurement_name != None:
                name = self.fringe_measurement_name
            self._last_measurement_fringe = sys.get_measurements(
                self._mirror_measure_point, self._mirror_measure_distance, name)

        # update statuses
        self._processing_fringes = False

    def close_all(self):
        """Closes all cameras, projectors, and sofast systems (currently just sofast fringe)"""
        with et.ignored(Exception):
            lt.debug("ServerState.close_all(): Stopping processing thread(s)")
            self._processing_pool.shutdown(wait=True, cancel_futures=True)
            self._processing_pool = None

        with et.ignored(Exception):
            lt.debug("ServerState.close_all(): Closing the cameras")
            for camera in ImageAcquisitionAbstract.instances():
                with et.ignored(Exception):
                    camera.close()

        with et.ignored(Exception):
            lt.debug("ServerState.close_all(): Closing the projector")
            ImageProjection.instance().close()

        with et.ignored(Exception):
            lt.debug("ServerState.close_all(): Closing the sofast fringe system")
            if self._system_fringe is not None:
                with self._system_fringe as sys:
                    sys.close_all()
                    self._system_fringe = None

        # TODO uncomment in the case that sofast fixed ever gains a close() or close_all() method
        # with et.ignored(Exception):
        #     lt.debug("ServerState.close_all(): Closing the sofast fixed system")
        #     if self._system_fixed is not None:
        #         with self._system_fixed as sys:
        #             sys.close_all()
        #             self._system_fixed = None

        with et.ignored(Exception):
            lt.debug("ServerState.close_all(): Unassigning this instance as the singleton ServerState")
            if self == ServerState._instance:
                ServerState._instance = None
