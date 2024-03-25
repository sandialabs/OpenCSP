"""Class for controlling displaying Sofast patterns and capturing images
"""

import datetime as dt
from typing import Callable
from warnings import warn

from numpy import ndarray
import numpy as np

from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe as Measurement
from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.geometry.Vxyz import Vxyz


class SystemSofastFringe:
    def __init__(
        self,
        image_projection: ImageProjection,
        image_acquisition: ImageAcquisitionAbstract | list[ImageAcquisitionAbstract],
    ) -> 'SystemSofastFringe':
        """
        Instantiates SystemSofastFringe class.

        Parameters
        ----------
        image_projection : ImageProjection
            Loaded ImageProjection object.
        image_acquisition : ImageAcquisition | List[ImageAcquisition, ...]
            Loaded ImageAcquisition object.

        """
        # Save objects in class
        self.root = image_projection.root

        self.image_projection = image_projection
        if isinstance(image_acquisition, list) and isinstance(image_acquisition[0], ImageAcquisitionAbstract):
            self.image_acquisition = image_acquisition
        elif isinstance(image_acquisition, ImageAcquisitionAbstract):
            self.image_acquisition = [image_acquisition]
        else:
            raise TypeError(f'ImageAcquisition must be instance or list of type {ImageAcquisitionAbstract}.')

        # Show crosshairs
        self.image_projection.show_crosshairs()

        # Instantiate all measurement data
        self.reset_all_measurement_data()

        # Create mask images to display
        self._create_mask_images_to_display()

        # Initialize queue
        self._queue_funcs = []

        # Define variables
        self.fringes: Fringes
        self.fringe_images_to_display: list[ndarray]
        self.mask_images_captured: list[ndarray]
        self.fringe_images_captured: list[ndarray]
        self.calibration_display_values: ndarray
        self.calibration_images: list[ndarray]

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
        self.fringes = None

        self.mask_images_to_display = None
        self.mask_images_captured = None

        self.fringe_images_to_display = None
        self.fringe_images_captured = None

        self.calibration_images = None
        self.calibration_display_values = None

    def _create_mask_images_to_display(self) -> None:
        """
        Creates black and white (in that order) images in active area for
        capturing mask of mirror.

        """
        self.mask_images_to_display = []

        # Create black image
        array = np.zeros(
            (self.image_projection.size_y, self.image_projection.size_x, 3),
            dtype=self.image_projection.display_data['projector_data_type'],
        )
        self.mask_images_to_display.append(array)

        # Create white image
        array = (
            np.zeros(
                (self.image_projection.size_y, self.image_projection.size_x, 3),
                dtype=self.image_projection.display_data['projector_data_type'],
            )
            + self.image_projection.max_int
        )
        self.mask_images_to_display.append(array)

    def _measure_sequence_display(
        self, im_disp_list: list, im_cap_list: list[list[ndarray]], run_next: Callable | None = None
    ) -> None:
        """
        Displays next image in sequence, waits, then captures frame from camera

        Parameters
        ----------
        im_disp_list : list[ndarray]
            3D RGB images to display.
        im_cap_list : list[list[ndarray, ...], ...]
            2D images captured by camera.
        run_next : Callable
            Function that is run after all images have been captured.

        """
        # Display image
        frame_idx = len(im_cap_list[0])
        self.image_projection.display_image_in_active_area(im_disp_list[frame_idx])

        # Wait, then capture image
        self.root.after(
            self.image_projection.display_data['image_delay'],
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
        for ims_cap, im_aq in zip(im_cap_list, self.image_acquisition):
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

    def load_fringes(self, fringes: Fringes, min_display_value: int) -> None:
        """
        Loads fringe object and creates RGB fringe images to display.

        Parameters
        ----------
        fringes : Fringes
            Fringe object to display.
        min_display_value : int
            Minimum display value to project.

        """
        # Save fringes
        self.fringes = fringes

        # Get fringe range
        fringe_range = (min_display_value, self.image_projection.display_data['projector_max_int'])

        # Get fringe base images
        fringe_images_base = fringes.get_frames(
            self.image_projection.size_x,
            self.image_projection.size_y,
            self.image_projection.display_data['projector_data_type'],
            fringe_range,
        )

        # Create full fringe display images
        self.fringe_images_to_display = []
        for idx in range(fringe_images_base.shape[2]):
            # Create image
            self.fringe_images_to_display.append(np.concatenate([fringe_images_base[:, :, idx : idx + 1]] * 3, axis=2))

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
            warn('Image is {:.2f}% saturated.'.format(saturation * 100), stacklevel=2)

    def capture_mask_images(self, run_next: Callable | None = None) -> None:
        """
        Captures mask images only. When finished, "run_next" is called. Images
        are stored in "mask_images_captured"

        Parameters
        ----------
        run_next : Callable, optional
            Function that is called after all images are captured. The default
            is self.close_all().

        """
        # Initialize mask image list
        self.mask_images_captured = []
        for _ in range(len(self.image_acquisition)):
            self.mask_images_captured.append([])

        # Start capturing images
        self._measure_sequence_display(self.mask_images_to_display, self.mask_images_captured, run_next)

    def capture_fringe_images(self, run_next: Callable | None = None) -> None:
        """
        Captures fringe images only. When finished, "run_next" is called.
        Images are stored in "fringe_images_captured"

        Parameters
        ----------
        run_next : Callable
            Function that is called after all images are captured. The default
            is self.close_all().

        """
        # Check fringes/camera have been loaded
        if self.fringes is None:
            raise ValueError('Fringes have not been loaded.')

        # Initialize fringe image list
        self.fringe_images_captured = []
        for _ in range(len(self.image_acquisition)):
            self.fringe_images_captured.append([])

        # Start capturing images
        self._measure_sequence_display(self.fringe_images_to_display, self.fringe_images_captured, run_next)

    def capture_mask_and_fringe_images(self, run_next: Callable | None = None) -> None:
        """
        Captures mask frames, then captures fringe images.

        Mask and fringe images are stored in:
            "mask_images_captured"
            "fringe_images_captured"

        Parameters
        ----------
        run_next : Callable
            Function that is called after all images are captured. The default
            is self.close_all().

        """
        # Check fringes/camera have been loaded
        if self.fringes is None:
            raise ValueError('Fringes have not been loaded.')

        def run_after_capture():
            self.capture_fringe_images(run_next)

        # Capture mask images, then capture fringe images, then run_next
        self.capture_mask_images(run_after_capture)

    def run_display_camera_response_calibration(self, res: int = 10, run_next: Callable | None = None) -> None:
        """
        Calculates camera-projector response data. Data is saved in
        calibration_display_values and calibration_images.

        Parameters
        ----------
        res : int, optional
            Digital number step size when stepping from 0 to
            "projector_max_int".
        run_next : Callable
            Process to run after calibration is performed.

        """
        # Generate grayscale values
        self.calibration_display_values = np.arange(
            0, self.image_projection.max_int + 1, res, dtype=self.image_projection.display_data['projector_data_type']
        )
        if self.calibration_display_values[-1] != self.image_projection.max_int:
            self.calibration_display_values = np.concatenate(
                (self.calibration_display_values, [self.image_projection.max_int])
            )

        # Generate grayscale images
        cal_images_display = []
        for dn in self.calibration_display_values:
            # Create image
            array = (
                np.zeros(
                    (self.image_projection.size_y, self.image_projection.size_x, 3),
                    dtype=self.image_projection.display_data['projector_data_type'],
                )
                + dn
            )
            cal_images_display.append(array)

        # Capture calibration images
        self.calibration_images = []
        for _ in range(len(self.image_acquisition)):
            self.calibration_images.append([])
        self._measure_sequence_display(cal_images_display, self.calibration_images, run_next)

    def run_camera_exposure_calibration(self, run_next: Callable | None = None) -> None:
        """
        Calculates the ideal camera exposure as to not saturate above a
        specified threshold.

        Parameters
        ----------
        run_next : Callable
            Process to run after calibration is performed.

        """

        def run_cal():
            # Calibrate exposure
            for im_aq in self.image_acquisition:
                im_aq.calibrate_exposure()

            # Run next operation
            if run_next is not None:
                run_next()

        # Set displayed image to white and calibrate exposure
        self.image_projection.display_image_in_active_area(self.mask_images_to_display[1])
        self.root.after(100, run_cal)

    def get_calibration_images(self) -> list[ndarray]:
        """
        Returns list of sets of calibration images as 3D ndarrays.

        Returns
        -------
        images : list[ndarray, ...]
            List of shape (N, M, n) arrays.

        """
        if self.calibration_images is None:
            raise ValueError('Calibration Images have not been collected yet.')

        images = []
        for ims in self.calibration_images:
            images.append(np.concatenate(ims, axis=2))
        return images

    def get_measurements(self, v_measure_point: Vxyz, optic_screen_dist: float, name: str) -> list[Measurement]:
        """
        Returns measurement object once mask and fringe images have been
        captured.

        Parameters
        ----------
        v_measure_point : Vxyz
            Location of measure point in optic coordinates.
        optic_screen_dist : float
            Distance from mirror to center of screen during measurement.
        name : str
            Name/serial number of measurement.

        Returns
        -------
        list[Measurement]
            Output list of measurement objects.

        """
        # Check data has been captured
        if self.fringe_images_captured is None:
            raise ValueError('Fringe images have not been captured.')
        if self.mask_images_captured is None:
            raise ValueError('Mask images have not been captured.')

        measurements = []
        for fringe_images, mask_images in zip(self.fringe_images_captured, self.mask_images_captured):
            # Create measurement object
            kwargs = dict(
                fringe_periods_x=np.array(self.fringes.periods_x),
                fringe_periods_y=np.array(self.fringes.periods_y),
                fringe_images=np.concatenate(fringe_images, axis=2),
                mask_images=np.concatenate(mask_images, axis=2),
                measure_point=v_measure_point,
                optic_screen_dist=optic_screen_dist,
                date=dt.datetime.now(),
                name=name,
            )
            measurements.append(Measurement(**kwargs))

        return measurements

    def close_all(self):
        """Closes all windows"""
        # Close image acquisition
        for im_aq in self.image_acquisition:
            im_aq.close()

        # Close window
        self.root.destroy()

    def set_queue(self, funcs: list[Callable]) -> None:
        """Sets the queue to given list of Callables"""
        self._queue_funcs = funcs

    def run_next_in_queue(self) -> None:
        """Runs the next funtion in the queue and removes from queue"""
        if len(self._queue_funcs) > 0:
            func = self._queue_funcs.pop(0)
            func()
