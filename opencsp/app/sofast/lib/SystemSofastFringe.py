import copy
import datetime as dt
from typing import Callable

from numpy import ndarray
import numpy as np

from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.app.sofast.lib.ImageCalibrationAbstract import ImageCalibrationAbstract
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe as Measurement
import opencsp.app.sofast.lib.DistanceOpticScreen as osd
from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.hdf5_tools as h5
import opencsp.common.lib.tool.log_tools as lt


class SystemSofastFringe:
    """Class for controlling/displaying Sofast patterns and capturing images"""

    def __init__(
        self, image_acquisition: ImageAcquisitionAbstract | list[ImageAcquisitionAbstract] = None
    ) -> "SystemSofastFringe":
        """
        Instantiates SystemSofastFringe class.

        Parameters
        ----------
        image_acquisition : ImageAcquisition | List[ImageAcquisition, ...], optional
            Loaded ImageAcquisition object, for verification of existance and registration in internal list of cameras.
            These acquisitions are tracked, so that as they are closed this instance releases its reference to them. If
            all acquisitions are closed at the time when a method is executed, then we fall back on the global instance.

        Raises
        ------
        RuntimeError:
            Either the image_acquisition or the image_projection has not been loaded.
        TypeError:
            The image_acquisition is not the correct type.
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
        if isinstance(image_acquisition, list):
            self._image_acquisitions = image_acquisition
        else:
            self._image_acquisitions = [image_acquisition]

        # Save objects in class
        self.root = image_projection.root

        # Show crosshairs
        image_projection.show_crosshairs()

        # Instantiate all measurement data
        self.reset_all_measurement_data()

        # Create mask images to display
        self._create_mask_images_to_display()

        # Initialize queue
        self._queue_funcs = []

        # Define variables
        self.fringes: Fringes = None
        """Fringe object used to create fringe images to display"""
        self._fringe_images_to_display: list[ndarray]
        """List of 2d fringe images to display"""
        self._mask_images_to_display: list[ndarray]
        """List of 2d mask images to display"""
        self._mask_images_captured: list[list[ndarray]]
        """List of lists of captured mask images (one list per camera)"""
        self._fringe_images_captured: list[list[ndarray]]
        """List of lists of images captured during run_measurement. Outer index is the camera index. Inner index is the fringe image."""
        self._calibration_display_values: ndarray
        """The digital numbers sent to the projector for the projector-camera response calibration"""
        self._calibration_images_captured: list[list[ndarray]]
        """List of lists of projector-camera response calibration images (one list per camera)"""
        self._calibration: ImageCalibrationAbstract = None
        """The computed ImageCalibration object"""

    def __del__(self):
        # Remove references to this instance from the ImageAcquisition cameras
        image_acquisitions: list[ImageAcquisitionAbstract] = []
        with et.ignored(Exception):
            image_acquisitions = self._image_acquisitions
        for ia in image_acquisitions:
            with et.ignored(Exception):
                ia.on_close.remove(self._on_image_acquisition_close)

        # Close Fringe-specific objects
        self.reset_all_measurement_data()

    def set_fringes(self, fringes: Fringes) -> None:
        """Sets the fringes object in the class

        Parameters
        ----------
        fringes : Fringes
            Fringe object to set to System
        """
        self.fringes = fringes

        if self._calibration is not None:
            self.create_fringe_images_from_image_calibration()

    def set_calibration(self, calibration: ImageCalibrationAbstract) -> None:
        """
        Loads calibration object and creates RGB fringe images to display.

        Parameters
        ----------
        calibration : ImageCalibrationAbstract
            The image calibration object to use during fringe image generation
        """
        self._calibration = calibration

        if self.fringes is not None:
            self.create_fringe_images_from_image_calibration()

    @property
    def image_acquisitions(self) -> list[ImageAcquisitionAbstract]:
        """Loaded image acquisition instances"""
        if len(self._image_acquisitions) == 0:
            # Import here to avoide circular dependencies
            import opencsp.app.sofast.lib.sofast_common_functions as scf

            # Check for existance
            scf.check_camera_loaded("SystemSofastFringe")
            # Use global instance as a backup, in case all previously registered instances have been closed
            self._image_acquisitions = [ImageAcquisitionAbstract.instance()]
        return self._image_acquisitions

    @image_acquisitions.setter
    def image_acquisitions(self, image_acquisitions: list[ImageAcquisitionAbstract]):
        # Validate input
        if (
            isinstance(image_acquisitions, list)
            and len(image_acquisitions) > 0
            and isinstance(image_acquisitions[0], ImageAcquisitionAbstract)
        ):
            pass
        elif isinstance(image_acquisitions, ImageAcquisitionAbstract):
            image_acquisitions = [image_acquisitions]
        else:
            lt.error_and_raise(
                TypeError,
                f"Error in SystemSofastFringe(): ImageAcquisition must be instance or list of type {ImageAcquisitionAbstract}.",
            )

        # Set value
        self._image_acquisitions = image_acquisitions

        # Register on_close handler
        for ia in self.image_acquisitions:
            ia.on_close.append(self._on_image_acquisition_close)

    def _on_image_acquisition_close(self, image_acquisition: ImageAcquisitionAbstract):
        self._image_acquisitions.remove(image_acquisition)

    def run(self) -> None:
        """
        Instantiates the system by starting the mainloop of the ImageProjection
        object.

        """
        self.run_next_in_queue()
        self.root.mainloop()

    def reset_all_measurement_data(self) -> None:
        """
        Resets all attributes related to capturing measurements

        """
        # Reset measurement attributes to None
        self._mask_images_to_display = None
        self._mask_images_captured = None

        self._fringe_images_to_display = None
        self._fringe_images_captured = None

        self._calibration_images_captured = None
        self._calibration_display_values = None
        self._calibration = None

    def _create_mask_images_to_display(self) -> None:
        """
        Creates black and white (in that order) images in active area for
        capturing mask of mirror.
        """
        self._mask_images_to_display = []
        image_projection = ImageProjection.instance()
        # Create black image
        array = np.array(image_projection.get_black_array_active_area())
        self._mask_images_to_display.append(array)
        # Create white image
        array = (
            np.array(image_projection.get_black_array_active_area()) + image_projection.display_data.projector_max_int
        )
        self._mask_images_to_display.append(array)

    def _measure_sequence_display(
        self, im_disp_list: list, im_cap_list: list[list[ndarray]], run_next: Callable | None = None
    ) -> None:
        """
        Displays next image in sequence, waits, then captures frame from camera.

        Parameters
        ----------
        im_disp_list : list[ndarray]
            3D RGB images to display.
        im_cap_list : list[list[ndarray, ...], ...]
            2D images captured by camera.
        run_next : Callable
            Function that is run after all images have been captured.

        """
        image_projection = ImageProjection.instance()

        # Display image
        frame_idx = len(im_cap_list[0])
        image_projection.display_image_in_active_area(im_disp_list[frame_idx])

        # Wait, then capture image
        self.root.after(
            image_projection.display_data.image_delay_ms,
            lambda: self._measure_sequence_capture(im_disp_list, im_cap_list, run_next),
        )

    def _measure_sequence_capture(
        self, im_disp_list: list, im_cap_list: list[list], run_next: Callable | None = None
    ) -> None:
        """
        Captures image from camera. If more images to display, loops to
        display next image. Otherwise, executes "run_next"

        Parameters
        ----------
        im_disp_list : list[ndarray]
            3D RGB images to display.
        im_cap_list : list[list[ndarray, ...], ...]
            2D images captured by camera.
        run_next : Callable
            Function that is run after all images have been captured.

        """
        for ims_cap, im_aq in zip(im_cap_list, self.image_acquisitions):
            # Capture and save image
            im = im_aq.get_frame()

            # Check for image saturation
            self.check_saturation(im, im_aq.max_value)

            # Reshape image if necessary
            if np.ndim(im) == 2:
                im = im[..., np.newaxis]

            # Save image to image list
            ims_cap.append(im)

        if len(im_cap_list[0]) < len(im_disp_list):
            # Display next image if not finished
            self.root.after(10, lambda: self._measure_sequence_display(im_disp_list, im_cap_list, run_next))
        elif run_next is not None:
            # Run next operation if finished
            run_next()

    @property
    def calibration(self):
        """Returns ImageCalibration object. Note: this is not a setter, use set_calibration() instead"""
        return self._calibration

    def create_fringe_images_from_image_calibration(self):
        """Once the system loads a new calibration object, a new min_display_value and
        RGB fringe images must be made
        """
        # Check fringes are loaded
        if self.fringes is None:
            lt.error_and_raise(
                ValueError,
                "Error in SystemSofastFringe.create_fringe_images_from_image_calibration(): "
                + "fringes must be set using self.set_frignes() first.",
            )

        # Check calibration is loaded
        if self._calibration is None:
            lt.error_and_raise(
                ValueError,
                "Error in SystemSofastFringe.create_fringe_images_from_image_calibration"
                + "calibration must be set using self.set_calibration first.",
            )

        image_projection = ImageProjection.instance()

        # Calculate minimun display value
        min_display_value = self._calibration.calculate_min_display_camera_values()[0]

        # Get fringe range
        fringe_range = (min_display_value, image_projection.display_data.projector_max_int)

        # Get fringe base images
        fringe_images_base = self.fringes.get_frames(
            image_projection.display_data.active_area_size_x,
            image_projection.display_data.active_area_size_y,
            image_projection.display_data.projector_data_type,
            fringe_range,
        )

        # Create full fringe display images
        self._fringe_images_to_display = []
        for idx in range(fringe_images_base.shape[2]):
            # Create image
            self._fringe_images_to_display.append(np.concatenate([fringe_images_base[:, :, idx : idx + 1]] * 3, axis=2))

    def check_saturation(self, image: ndarray, camera_max_int: int, thresh: float = 0.005) -> None:
        """
        Checks if input image is saturated. Gives warning if image is saturated
        above given threshold.

        Parameters
        ----------
        image : ndarray
            Input image.
        camera_max_int : int
            Saturation value of camera.
        thresh : float, optional
            Fraction of image saturation that is acceptable. The default is
            0.5%.

        """
        # Calculate fraction of image saturated
        saturation = (image >= camera_max_int).sum() / image.size  # fraction

        # Compare to threshold
        if saturation >= thresh:
            lt.warn(f"Image is {saturation * 100:.2f}% saturated.")

    def run_gray_levels_cal(
        self,
        calibration_class: type[ImageCalibrationAbstract],
        calibration_hdf5_path_name_ext: str = None,
        on_capturing: Callable = None,
        on_captured: Callable = None,
        on_processing: Callable = None,
        on_processed: Callable = None,
    ) -> None:
        """Runs the projector-camera intensity calibration and stores the results in self._calibration_images_captured and
        self._calibration.

        Params:
        -------
        calibration_class : type[ImageCalibrationAbstract]:
            The type of calibration to use.
        calibration_hdf5_path_name_ext : str, optional
            The pathname of the output HDF5 file, does not save if None. Default is None
        on_capturing : Callable, optional
            The callback to execute when capturing is about to start. Default is None
        on_captured : Callable, optional
            The callback to execute when capturing has finished. Default is None
        on_processing : Callable, optional
            The callback to execute when processing is about to start. Default is None
        on_processed : Callable, optional
            The callback to execute when processing has finished. At the point this callback is being called the
            calibration attribute will be set. Default is None
        """

        # Capture images
        def _func_0():
            # Run the "on capturing" callback
            if on_capturing != None:
                on_capturing()

            self.run_display_camera_response_calibration(res=10, run_next=self.run_next_in_queue)

            # Run the "on captured" callback
            if on_captured != None:
                on_captured()

        # Process data
        def _func_1():
            # Run the "on processing" callback
            if on_processing != None:
                on_processing()

            # Get calibration images from System
            calibration_images = self.get_calibration_images()[0]  # only calibrating one camera
            # Set calibration object
            self.set_calibration(calibration_class.from_data(calibration_images, self._calibration_display_values))
            # Save calibration object
            if calibration_hdf5_path_name_ext != None:
                self._calibration.save_to_hdf(calibration_hdf5_path_name_ext)
            # Save calibration raw data
            if calibration_hdf5_path_name_ext != None:
                data = [self._calibration.display_values, calibration_images]
                datasets = ["CalibrationRawData/display_values", "CalibrationRawData/images"]
                h5.save_hdf5_datasets(data, datasets, calibration_hdf5_path_name_ext)
            # Run the "on done" callback
            if on_processed != None:
                on_processed()
            # Continue
            self.run_next_in_queue()

        self.set_queue([_func_0, _func_1])
        self.run_next_in_queue()

    def capture_mask_images(self, run_next: Callable | None = None) -> None:
        """
        Captures mask images only. When finished, "run_next" is called. Images
        are stored in "_mask_images_captured"

        Parameters
        ----------
        run_next : Callable, optional
            Function that is called after all images are captured. The default
            is self.close_all(). TODO is this still the default?

        """
        # Initialize mask image list
        self._mask_images_captured = []
        for _ in range(len(self.image_acquisitions)):
            self._mask_images_captured.append([])

        # Start capturing images
        self._measure_sequence_display(self._mask_images_to_display, self._mask_images_captured, run_next)

    def capture_fringe_images(self, run_next: Callable | None = None) -> None:
        """
        Captures fringe images only. When finished, "run_next" is called.
        Images are stored in "_fringe_images_captured"

        Parameters
        ----------
        run_next : Callable
            Function that is called after all images are captured. The default
            is self.close_all(). TODO is this still the default?

        """
        # Check fringes/camera have been loaded
        if self.fringes is None:
            lt.error_and_raise(
                ValueError, "Error in SystemSofastFringe.capture_fringe_images(): Fringes have not been loaded."
            )

        if self._fringe_images_to_display is None:
            lt.error_and_raise(
                ValueError, "Error in SystemSofastFrigne.capture_fringe_images(): Fringe images have not been created"
            )

        # Initialize fringe image list
        self._fringe_images_captured = []
        for _ in range(len(self.image_acquisitions)):
            self._fringe_images_captured.append([])

        # Start capturing images
        self._measure_sequence_display(self._fringe_images_to_display, self._fringe_images_captured, run_next)

    def capture_mask_and_fringe_images(self, run_next: Callable | None = None) -> None:
        """
        Captures mask frames, then captures fringe images.

        Mask and fringe images are stored in:
            "_mask_images_captured"
            "_fringe_images_captured"

        Parameters
        ----------
        run_next : Callable
            Function that is called after all images are captured. The default
            is self.close_all(). TODO is this still the default?

        """
        # Check fringes/camera have been loaded
        if self.fringes is None:
            lt.error_and_raise(
                ValueError,
                "Error in SystemSofastFringe.capture_mask_and_fringe_images(): Fringes have not been loaded.",
            )

        def run_after_capture():
            self.capture_fringe_images(run_next)

        # Capture mask images, then capture fringe images, then run_next
        self.capture_mask_images(run_after_capture)

    def run_measurement(self, on_done: Callable = None) -> None:
        """Runs data collection with the given fringes.

        Once the data has been captured, the images will be available from this instance. Similarly, the data can then
        be processed, such as with get_measurements().

        Params
        ------
        on_done: Callable
            The function to call when capturing fringe images has finished.

        Raises
        ------
        RuntimeError:
            Calibration hasn't been set, or fringes haven't been set
        """
        # Get minimum display value from calibration
        if self._calibration is None:
            lt.error_and_raise(
                RuntimeError,
                "Error in SystemSofastFringe.run_measurement(): must have run or provided a calibration before starting a measurement.",
            )
        if self.fringes is None:
            lt.error_and_raise(
                RuntimeError,
                "Error in SystemSofastFringe.run_measurement(): must have set fringes before starting a measurement.",
            )

        # Capture images
        self.capture_mask_and_fringe_images(on_done)

    def run_display_camera_response_calibration(self, res: int = 10, run_next: Callable | None = None) -> None:
        """
        Calculates camera-projector response data. Data is saved in
        _calibration_display_values and _calibration_images_captured.

        Parameters
        ----------
        res : int, optional
            Digital number step size when stepping from 0 to
            "projector_max_int".
        run_next : Callable
            Process to run after calibration is performed.

        """
        image_projection = ImageProjection.instance()

        # Generate grayscale values
        self._calibration_display_values = np.arange(
            0,
            image_projection.display_data.projector_max_int + 1,
            res,
            dtype=image_projection.display_data.projector_data_type,
        )
        if self._calibration_display_values[-1] != image_projection.display_data.projector_max_int:
            self._calibration_display_values = np.concatenate(
                (self._calibration_display_values, [image_projection.display_data.projector_max_int])
            )

        # Generate grayscale images
        cal_images_display = []
        for dn in self._calibration_display_values:
            # Create image
            array = (
                np.zeros(
                    (
                        image_projection.display_data.active_area_size_y,
                        image_projection.display_data.active_area_size_x,
                        3,
                    ),
                    dtype=image_projection.display_data.projector_data_type,
                )
                + dn
            )
            cal_images_display.append(array)

        # Capture calibration images
        self._calibration_images_captured = []
        for _ in range(len(self.image_acquisitions)):
            self._calibration_images_captured.append([])
        self._measure_sequence_display(cal_images_display, self._calibration_images_captured, run_next)

    def get_calibration_images(self) -> list[ndarray]:
        """
        Returns list of sets of calibration images as 3D ndarrays.

        Returns
        -------
        images : list[ndarray, ...]
            List of shape (N, M, n) arrays.

        """
        if self._calibration_images_captured is None:
            lt.error_and_raise(
                ValueError,
                "Error in SystemSofastFringe.get_calibration_images(): Calibration Images have not been collected yet.",
            )

        images = []
        for ims in self._calibration_images_captured:
            images.append(np.concatenate(ims, axis=2))
        return images

    def get_measurements(self, v_measure_point: Vxyz, dist_optic_screen: float, name: str) -> list[Measurement]:
        """
        Returns measurement object once mask and fringe images have been
        captured.

        Parameters
        ----------
        v_measure_point : Vxyz
            Location of measure point in optic coordinates.
        dist_optic_screen : float
            Distance from mirror to center of screen during measurement.
        name : str
            Name/serial number of measurement.

        Returns
        -------
        list[Measurement]
            Output list of measurement objects.

        """
        # Check data has been captured
        if self._fringe_images_captured is None:
            lt.error_and_raise(
                ValueError, "Error in SystemSofastFringe.get_measurements(): Fringe images have not been captured."
            )
        if self._mask_images_captured is None:
            lt.error_and_raise(
                ValueError, "Error in SystemSofastFringe.get_measurements(): Mask images have not been captured."
            )

        measurements = []
        for fringe_images, mask_images in zip(self._fringe_images_captured, self._mask_images_captured):
            # Create measurement object
            dist_optic_screen_measure = osd.DistanceOpticScreen(v_measure_point, dist_optic_screen)
            kwargs = dict(
                fringe_periods_x=np.array(self.fringes.periods_x),
                fringe_periods_y=np.array(self.fringes.periods_y),
                fringe_images=np.concatenate(fringe_images, axis=2),
                mask_images=np.concatenate(mask_images, axis=2),
                dist_optic_screen_measure=dist_optic_screen_measure,
                date=dt.datetime.now(),
                name=name,
            )
            measurements.append(Measurement(**kwargs))

        return measurements

    def close_all(self):
        """Closes all windows"""
        # Close image acquisitions
        for im_aq in copy.copy(self.image_acquisitions):
            with et.ignored(Exception):
                im_aq.close()

        # Close window
        with et.ignored(Exception):
            self.root.destroy()

    def set_queue(self, funcs: list[Callable]) -> None:
        """Sets the queue to given list of Callables"""
        self._queue_funcs = funcs

    def run_next_in_queue(self) -> None:
        """Runs the next funtion in the queue and removes from queue"""
        if len(self._queue_funcs) > 0:
            func = self._queue_funcs.pop(0)
            func()
