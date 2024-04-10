import asyncio

import opencsp.app.sofast.lib.Executor as sfe
from opencsp.app.sofast.lib.SystemSofastFringe import SystemSofastFringe
from opencsp.app.sofast.lib.SystemSofastFixed import SystemSofastFixed
from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
import opencsp.common.lib.geometry.Vxyz as vxyz
import opencsp.common.lib.process.ControlledContext as cc
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.log_tools as lt


class ServerState:
    _instance: cc.ControlledContext['ServerState'] = None

    def __init__(self):
        # systems
        self._system_fixed: cc.ControlledContext[SystemSofastFixed] = None
        self._system_fringe: cc.ControlledContext[SystemSofastFringe] = None

        # measurements
        self._last_measurement_fixed: sfe.FixedResults = None
        self._last_measurement_fringe: sfe.FringeResults = None
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

        # processing manager
        self._executor = sfe.Executor()

        # don't try to close more than once
        self.is_closed = False

        # assign this as the global static instance
        if ServerState._instance is None:
            ServerState._instance = cc.ControlledContext(self)
        else:
            lt.error_and_raise(
                RuntimeError,
                "Error in ServerState(): "
                + "this class is supposed to be a singleton, but another instance already exists!",
            )

    def __del__(self):
        with et.ignored(Exception):
            self.close_all()

    @staticmethod
    def instance() -> cc.ControlledContext['ServerState']:
        return ServerState._instance

    @property
    @staticmethod
    def _state_lock() -> asyncio.Lock:
        """
        Return the mutex for providing exclusive access to ServerState singleton.

        I (BGB) use "with instance()" when calling state methods from code external to the state class, and "with
        _state_lock()" when modifying critical sections of code within the state class (mostly for thread safety in
        regards to the processing thread).  However, the following statements are equivalent and can be used
        interchangeably::

            # 1. using the instance() method
            with self.instance() as state:
                # do stuff

            # 2. using the _state_lock() method
            with self._state_lock:
                # do stuff
        """
        return ServerState.instance().mutex

    @property
    def system_fixed(self) -> cc.ControlledContext[SystemSofastFixed]:
        if self._system_fixed is None:
            display_data = ImageProjection.instance().display_data
            size_x, size_y = display_data['size_x'], display_data['size_y']
            width_pattern = self._fixed_pattern_diameter
            spacing_pattern = self._fixed_pattern_spacing
            self._system_fixed = cc.ControlledContext(SystemSofastFixed(size_x, size_y, width_pattern, spacing_pattern))
        return self._system_fixed

    @property
    def system_fringe(self) -> cc.ControlledContext[SystemSofastFringe]:
        if self._system_fringe is None:
            self._system_fringe = cc.ControlledContext(SystemSofastFringe())
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
            lt.error(
                "Programmer error in ServerState.fixed_measurement_ready(): "
                + f"expected 'last_measurement' to be None while 'running', but {self._last_measurement_fixed=} {self._running_measurement_fixed}"
            )
            return False
        elif self._processing_measurement_fixed:
            lt.error(
                "Programmer error in ServerState.fixed_measurement_ready(): "
                + f"expected 'last_measurement' to be None while 'processing', but {self._last_measurement_fixed=} {self._processing_measurement_fixed}"
            )
            return False
        else:
            return True

    @property
    def last_measurement_fixed(self) -> sfe.FixedResults:
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
            lt.error(
                "Programmer error in ServerState.fringe_measurement_ready(): "
                + f"expected 'last_measurement' to be None while 'running', but {self._last_measurement_fringe=} {self._running_measurement_fringe}"
            )
            return False
        elif self._processing_measurement_fringe:
            lt.error(
                "Programmer error in ServerState.fringe_measurement_ready(): "
                + f"expected 'last_measurement' to be None while 'processing', but {self._last_measurement_fringe=} {self._processing_measurement_fringe}"
            )
            return False
        else:
            return True

    @property
    def last_measurement_fringe(self) -> sfe.FringeResults:
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

        Once this method is called it returns immediately without waiting for collection and processing to finish.
        self.has_fringe_measurement will be False during the period, and will transition to True once collection and
        processing have both finished.

        The collection is queued up for the main thread (aka the tkinter thread), and processing is done in
        another thread once collection has finished.

        Returns
        -------
        success: bool
            True if the measurement was able to be started. False if the system resources are busy.
        """
        # Check that system resources are available
        if self._running_measurement_fringe or self._processing_measurement_fringe:
            lt.warn(
                "Warning in server_api.run_measurment_fringe(): "
                + "Attempting to start another fringe measurement before the last fringe measurement has finished."
            )
            return False
        if not self.projector_available:
            lt.warn(
                "Warning in server_api.run_measurment_fringe(): "
                + "Attempting to start a fringe measurement while the projector is already in use."
            )
            return False

        # Latch the name value
        self.fringe_measurement_name = name

        # Update statuses
        # Critical section, these statuses updates need to be thread safe
        with self._state_lock:
            self._last_measurement_fringe = None
            self._running_measurement_fringe = True

        # Start the measurement
        self._executor.on_fringe_collected = self._on_fringe_collected
        self._executor.start_collect_fringe(self.system_fringe)

        return True

    def _on_fringe_collected(self):
        """
        Registers the change in state from having finished capturing fringe images, and starts processing.

        This method is evaluated in the main thread (aka the tkinter thread), and so certain critical sections of code
        are protected to ensure a consistent state is maintained.
        """
        lt.debug("ServerState: finished collecting fringes")
        if not self._running_measurement_fringe:
            lt.error(
                "Programmer error in server_api._on_collect_fringes_done(): "
                + "Did not expect for this method to be called while self._running_measurement_fringe was not True!"
            )

        # Update statuses
        # Critical section, these statuses updates need to be thread safe
        with self._state_lock:
            self._processing_measurement_fringe = True
            self._running_measurement_fringe = False

        # Start the processing
        self._executor.on_fringe_processed = self._on_fringe_processed
        self._executor.start_process_fringe(
            self.system_fringe,
            self.mirror_measure_point,
            self.mirror_measure_distance,
            self.orientation,
            self.camera,
            self.display,
            self.facet_data,
            self.surface,
            self.fringe_measurement_name,
            self.reference_facet,
        )

    def _on_fringe_processed(self, fringe_results: sfe.FringeResults):
        """
        Processes the fringe images captured during self.system_fringe.run_measurement() and stores the result to
        self._last_measurement_fringe.

        This method is evaluated in the _processing_pool thread, and so certain critical sections of code are protected
        to ensure a consistent state is maintained.
        """
        lt.debug("ServerState: finished processing fringes")
        if not self._processing_measurement_fringe:
            lt.error(
                "Programmer error in server_api._process_fringes(): "
                + "Did not expect for this method to be called while self._processing_measurement_fringe was not True!"
            )

        # update statuses
        # Critical section, these statuses updates need to be thread safe
        with self._state_lock:
            self._processing_fringes = False
            self._last_measurement_fringe = fringe_results

    def close_all(self):
        """Closes all cameras, projectors, and sofast systems (currently just sofast fringe)"""
        # don't try to close more than once
        if self.is_closed:
            return
        self.is_closed = True

        with et.ignored(Exception):
            lt.debug("ServerState.close_all(): Stopping processing thread")
            self._executor.close()
            self._executor = None

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
            self._system_fringe = None

        with et.ignored(Exception):
            lt.debug("ServerState.close_all(): Closing the sofast fixed system")
            # TODO uncomment in the case that sofast fixed ever gains a close() or close_all() method
            #     if self._system_fixed is not None:
            #         with self._system_fixed as sys:
            #             sys.close_all()
            #             self._system_fixed = None
            self._system_fixed = None

        with et.ignored(Exception):
            lt.debug("ServerState.close_all(): Unassigning this instance as the singleton ServerState")
            if self == ServerState._instance:
                ServerState._instance = None
