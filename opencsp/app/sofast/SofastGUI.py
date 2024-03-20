"""GUI to control SOFAST image projection, image acquisition,
and data capture. Can capture datasets and save to HDF format.
"""

import datetime as dt
import tkinter
from tkinter import messagebox, simpledialog
from tkinter.filedialog import askopenfilename, asksaveasfilename

import imageio
import matplotlib.pyplot as plt
import numpy as np

from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract
from opencsp.common.lib.camera.ImageAcquisition_DCAM_mono import (
    ImageAcquisition as ImageAcquisition_DCAM,
)
from opencsp.common.lib.camera.ImageAcquisition_DCAM_color import (
    ImageAcquisition as ImageAcquisition_DCAM_color,
)
from opencsp.common.lib.camera.ImageAcquisition_MSMF import (
    ImageAcquisition as ImageAcquisition_MSMF,
)
from opencsp.common.lib.camera.image_processing import highlight_saturation
from opencsp.common.lib.camera.LiveView import LiveView
from opencsp.app.sofast.lib.ImageCalibrationAbstract import ImageCalibrationAbstract
from opencsp.app.sofast.lib.ImageCalibrationGlobal import ImageCalibrationGlobal
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.SystemSofastFringe import SystemSofastFringe
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool import hdf5_tools
from opencsp.common.lib.tool.TkToolTip import TkToolTip


class SofastGUI:
    """Class that contains SOFAST GUI controls and methods"""

    def __init__(self) -> 'SofastGUI':
        """
        Instantiates GUI in new window
        """
        # Define camera objects to choose from
        self.cam_options = [  # Description of camera
            'DCAM Mono',
            'DCAM Color',
            'MSMF Mono',
        ]
        self.cam_objects = [  # Camera object to use
            ImageAcquisition_DCAM,
            ImageAcquisition_DCAM_color,
            ImageAcquisition_MSMF,
        ]

        # Define calibration objects to use
        self.cal_options = ['Global', 'Scaling']
        self.cal_objects = [ImageCalibrationGlobal, ImageCalibrationScaling]

        # Create tkinter object
        self.root = tkinter.Tk()

        # Set title
        self.root.title('SOFAST')

        # Set size of GUI
        self.root.geometry('600x640+200+100')

        # Add all buttons/widgets to window
        self._create_layout()

        # Set defaults
        self.image_projection: ImageProjection = None
        self.image_acquisition: ImageAcquisitionAbstract = None
        self.calibration: ImageCalibrationAbstract = None
        self.system: SystemSofastFringe = None

        # Disable buttons
        self._enable_btns()

        # Run window infinitely
        self.root.mainloop()

    def _get_frame(self) -> np.ndarray:
        """
        Captures frame from camera

        Returns
        -------
        frame : ndarray
            2D image from camera.

        """
        # Check camera is loaded
        if self.image_acquisition is None:
            messagebox.showerror('Error', 'Camera is not connected.')
            return

        # Get frame
        frame = self.image_acquisition.get_frame()

        return frame

    def _plot_hist(self, ax: plt.Axes, frame: np.ndarray) -> None:
        """
        Plots the image pixel level histogram on the given axes.

        Parameters
        ----------
        ax : matplotlib Axes
            Axes to plot histogram on.
        frame : np.ndarray
            2D image array on which histogram is computed.

        """
        ax.hist(frame.flatten(), histtype='step', bins=100, density=True)
        ax.grid()
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Counts')
        ax.set_title('Image Histogram')

    def _plot_image(
        self,
        ax: plt.Axes,
        image: np.ndarray,
        title: str = '',
        xlabel: str = '',
        ylabel: str = '',
    ) -> None:
        """
        Plots image on given axes.

        Parameters
        ----------
        ax : matplotlib Axes
            Axes on which to plot image.
        image : np.ndarray
            2D or 3D image array.
        title : str, optional
            Title of plot. The default is ''.
        xlabel : str, optional
            X axis label. The default is ''.
        ylabel : str, optional
            Y axis label. The default is ''.

        """
        # Show image and format axes
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def _create_layout(self) -> None:
        """Creates GUI widgets"""
        # System control label frame
        label_frame_load_system = tkinter.LabelFrame(self.root, text='Load Components')
        label_frame_load_system.grid(row=0, column=0, sticky='nesw', padx=5, pady=5)
        # Projection control label frame
        label_frame_projector = tkinter.LabelFrame(self.root, text='Projection Control')
        label_frame_projector.grid(row=1, column=0, sticky='nesw', padx=5, pady=5)
        # Camera control label frame
        label_frame_camera = tkinter.LabelFrame(self.root, text='Camera controls')
        label_frame_camera.grid(row=2, column=0, sticky='nesw', padx=5, pady=5)
        # Data capture label frame
        label_frame_run = tkinter.LabelFrame(self.root, text='Run Options')
        label_frame_run.grid(row=3, column=0, sticky='nesw', padx=5, pady=5)
        # Settings label frame
        label_frame_settings = tkinter.LabelFrame(self.root, text='Settings')
        label_frame_settings.grid(
            row=0, column=1, rowspan=3, sticky='nesw', padx=5, pady=5
        )

        # =============== First Column - Load components ===============
        r = 0
        # Connect camera button
        self.btn_load_image_acquisition = tkinter.Button(
            label_frame_load_system,
            text='Connect Camera',
            command=self.load_image_acquisition,
        )
        self.btn_load_image_acquisition.grid(
            row=r, column=0, pady=2, padx=2, sticky='nesw'
        )
        TkToolTip(
            self.btn_load_image_acquisition,
            'Select the type of camera in the dropdown menu to the right. Click to connect to the camera.',
        )
        r += 1

        # Load physical layout
        self.btn_load_image_projection = tkinter.Button(
            label_frame_load_system,
            text='Load ImageProjection',
            command=self.load_image_projection,
        )
        self.btn_load_image_projection.grid(
            row=r, column=0, pady=2, padx=2, sticky='nesw'
        )
        TkToolTip(
            self.btn_load_image_projection,
            'Select an ImageProjection HDF file to load and display.',
        )
        r += 1

        # =============== First Column - Projection controls ===============
        r = 0
        self.btn_show_cal_image = tkinter.Button(
            label_frame_projector,
            text='Show Calibration Image',
            command=self.show_calibration_image,
        )
        self.btn_show_cal_image.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        TkToolTip(
            self.btn_show_cal_image, 'Shows calibration image on projection window.'
        )
        r += 1

        self.btn_show_axes = tkinter.Button(
            label_frame_projector, text='Show Screen Axes', command=self.show_axes
        )
        self.btn_show_axes.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        TkToolTip(self.btn_show_axes, 'Shows screen axes on projection window.')
        r += 1

        self.btn_show_crosshairs = tkinter.Button(
            label_frame_projector, text='Show Crosshairs', command=self.show_crosshairs
        )
        self.btn_show_crosshairs.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        TkToolTip(self.btn_show_crosshairs, 'Shows crosshairs on projection window.')
        r += 1

        self.btn_close_projection = tkinter.Button(
            label_frame_projector,
            text='Close Display Window',
            command=self.close_projection_window,
        )
        self.btn_close_projection.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        TkToolTip(self.btn_close_projection, 'Close only projection window.')
        r += 1

        # =============== First Column - Camera controls ===============
        r = 0
        # Perform exposure calibration
        self.btn_exposure_cal = tkinter.Button(
            label_frame_camera,
            text='Calibrate Exposure Time',
            command=self.run_exposure_cal,
        )
        self.btn_exposure_cal.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        TkToolTip(
            self.btn_exposure_cal, 'Automatically performs camera exposure calibration.'
        )
        r += 1

        # Set camera exposure
        self.btn_set_exposure = tkinter.Button(
            label_frame_camera, text='Set Exposure Time', command=self.set_exposure
        )
        self.btn_set_exposure.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        TkToolTip(
            self.btn_set_exposure,
            'Set the camera exposure time value. The current value is displayed when clicked.',
        )
        r += 1

        # Show snapshot button
        self.btn_show_snapshot = tkinter.Button(
            label_frame_camera, text='Show Snapshot', command=self.show_snapshot
        )
        self.btn_show_snapshot.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        TkToolTip(
            self.btn_show_snapshot,
            'Shows a camera image and pixel brightness histogram.',
        )
        r += 1

        # Save snapshot button
        self.btn_save_snapshot = tkinter.Button(
            label_frame_camera, text='Save Snapshot', command=self.save_snapshot
        )
        self.btn_save_snapshot.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        TkToolTip(self.btn_save_snapshot, 'Saves image from camera as a PNG.')
        r += 1

        # Live view button
        self.btn_live_view = tkinter.Button(
            label_frame_camera, text='Live View', command=self.live_view
        )
        self.btn_live_view.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        TkToolTip(self.btn_live_view, 'Shows live view from connected camera.')
        r += 1

        # Perform exposure calibration
        self.btn_close_camera = tkinter.Button(
            label_frame_camera,
            text='Close Camera',
            command=self.close_image_acquisition,
        )
        self.btn_close_camera.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        TkToolTip(self.btn_close_camera, 'Closes connection to camera.')
        r += 1

        # =============== First Column - System run controls ===============
        r = 0
        # Run sofast capture
        self.btn_run_measurement = tkinter.Button(
            label_frame_run, text='Run Data Capture', command=self.run_measurement
        )
        self.btn_run_measurement.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        TkToolTip(
            self.btn_run_measurement,
            'Runs SOFAST data capture. Mask then fringes are captured.',
        )
        r += 1

        # Perform projector-camera brightness calibration
        self.btn_gray_levels_cal = tkinter.Button(
            label_frame_run,
            text='Run Response Calibration',
            command=self.run_gray_levels_cal,
        )
        self.btn_gray_levels_cal.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        TkToolTip(
            self.btn_gray_levels_cal,
            'Performs Projector-Camera Brightness Calibration sequence. Must select a destination HDF file first.',
        )
        r += 1

        # Load projector-camera brightness calibration
        self.btn_load_gray_levels_cal = tkinter.Button(
            label_frame_run,
            text='Load Response Calibration',
            command=self.load_gray_levels_cal,
        )
        self.btn_load_gray_levels_cal.grid(
            row=r, column=0, pady=2, padx=2, sticky='nesw'
        )
        TkToolTip(
            self.btn_load_gray_levels_cal,
            'Loads a previously saved Projector-Camera Brightness Calibration sequence data file.',
        )
        r += 1

        # View projector-camera brightness calibration
        self.btn_view_gray_levels_cal = tkinter.Button(
            label_frame_run,
            text='View Response Calibration',
            command=self.view_gray_levels_cal,
        )
        self.btn_view_gray_levels_cal.grid(
            row=r, column=0, pady=2, padx=2, sticky='nesw'
        )
        TkToolTip(
            self.btn_view_gray_levels_cal,
            'Views current Projector-Camera Brightness Calibration sequence data file.',
        )
        r += 1

        # =============== First Column - Close button ===============
        # Close window button
        self.btn_close = tkinter.Button(self.root, text='Close All', command=self.close)
        self.btn_close.grid(row=4, column=0, pady=2, padx=2, sticky='nesw')
        TkToolTip(self.btn_close, 'Closes all windows.')

        # =============== Second Column - Settings ===============
        r = 0
        # Camera type dropdown
        self.var_cam_select = tkinter.StringVar(value=self.cam_options[0])
        lbl_camera_type = tkinter.Label(
            label_frame_settings, text='Select camera:', font=('calibre', 10, 'bold')
        )
        drop_camera_type = tkinter.OptionMenu(
            label_frame_settings, self.var_cam_select, *self.cam_options
        )
        TkToolTip(drop_camera_type, 'Select type of camera object to load.')

        lbl_camera_type.grid(row=r, column=1, pady=2, padx=2, sticky='nse')
        drop_camera_type.grid(row=r, column=2, pady=2, padx=2, sticky='nsw')
        r += 1

        # Calibration type dropdown
        self.var_cal_select = tkinter.StringVar(value=self.cal_options[0])
        lbl_cal_type = tkinter.Label(
            label_frame_settings,
            text='Select calibration method:',
            font=('calibre', 10, 'bold'),
        )
        drop_cal_type = tkinter.OptionMenu(
            label_frame_settings, self.var_cal_select, *self.cal_options
        )
        TkToolTip(
            drop_cal_type,
            'Select type of Projector-Camera Brightness Calibration process to use.',
        )

        lbl_cal_type.grid(row=r, column=1, pady=2, padx=2, sticky='nse')
        drop_cal_type.grid(row=r, column=2, pady=2, padx=2, sticky='nsw')
        r += 1

        # Fringe periods input box
        self.var_fringe_periods_x = tkinter.IntVar(value=4)
        self.var_fringe_periods_y = tkinter.IntVar(value=4)
        lbl_fringe_x = tkinter.Label(
            label_frame_settings, text='Fringe X periods:', font=('calibre', 10, 'bold')
        )
        lbl_fringe_y = tkinter.Label(
            label_frame_settings, text='Fringe Y periods:', font=('calibre', 10, 'bold')
        )
        entry_fringe_x = tkinter.Spinbox(
            label_frame_settings,
            textvariable=self.var_fringe_periods_x,
            font=('calibre', 10, 'normal'),
            from_=1,
            to=100,
            width=5,
        )
        entry_fringe_y = tkinter.Spinbox(
            label_frame_settings,
            textvariable=self.var_fringe_periods_y,
            font=('calibre', 10, 'normal'),
            from_=1,
            to=100,
            width=5,
        )
        TkToolTip(
            entry_fringe_x,
            'Number of fringe X periods to show during a data capture. Each fringe period will always be phase shifted four times.',
        )
        TkToolTip(
            entry_fringe_y,
            'Number of fringe Y periods to show during a data capture. Each fringe period will always be phase shifted four times.',
        )

        lbl_fringe_x.grid(row=r, column=1, pady=2, padx=2, sticky='nse')
        entry_fringe_x.grid(row=r, column=2, pady=2, padx=2, sticky='nsw')
        r += 1
        lbl_fringe_y.grid(row=r, column=1, pady=2, padx=2, sticky='nse')
        entry_fringe_y.grid(row=r, column=2, pady=2, padx=2, sticky='nsw')
        r += 1

        # Camera calibration input box
        self.var_gray_lvl_cal_status = tkinter.StringVar(
            value='Calibration data: No Data'
        )
        lbl_gray_lvl_cal_status = tkinter.Label(
            label_frame_settings,
            textvariable=self.var_gray_lvl_cal_status,
            font=('calibre', 10, 'bold'),
        )

        lbl_gray_lvl_cal_status.grid(
            row=r, column=1, pady=2, padx=2, sticky='nsw', columnspan=2
        )
        r += 1

        # Measure point input
        self.var_meas_pt = tkinter.StringVar(value='0.0, 0.0, 0.0')
        lbl_meas_pt = tkinter.Label(
            label_frame_settings, text='Measure Point XYZ', font=('calibre', 10, 'bold')
        )
        entry_meas_pt = tkinter.Entry(
            label_frame_settings,
            textvariable=self.var_meas_pt,
            font=('calibre', 10, 'normal'),
            width=20,
        )
        TkToolTip(
            entry_meas_pt,
            'Location, in mirror coordinates (meters), of the point on the mirror where the "mirror-to-screen distance" is measured.',
        )

        lbl_meas_pt.grid(row=r, column=1, pady=2, padx=2, sticky='nsw', columnspan=2)
        r += 1
        entry_meas_pt.grid(row=r, column=1, pady=2, padx=2, sticky='nsw')
        r += 1

        # Distance input
        self.var_meas_dist = tkinter.StringVar(value='10.0')
        lbl_meas_dist = tkinter.Label(
            label_frame_settings,
            text='Measured mirror-screen distance',
            font=('calibre', 10, 'bold'),
        )
        entry_meas_dist = tkinter.Entry(
            label_frame_settings,
            textvariable=self.var_meas_dist,
            font=('calibre', 10, 'normal'),
            width=20,
        )
        TkToolTip(
            entry_meas_dist,
            'The distance in meters from the display crosshairs to the "Measure Point XYZ" location on the mirror.',
        )

        lbl_meas_dist.grid(row=r, column=1, pady=2, padx=2, sticky='nsw', columnspan=2)
        r += 1
        entry_meas_dist.grid(row=r, column=1, pady=2, padx=2, sticky='nsw')
        r += 1

        # Measurement name
        self.var_meas_name = tkinter.StringVar(value='')
        lbl_meas_name = tkinter.Label(
            label_frame_settings, text='Measurement name', font=('calibre', 10, 'bold')
        )
        entry_meas_name = tkinter.Entry(
            label_frame_settings,
            textvariable=self.var_meas_name,
            font=('calibre', 10, 'normal'),
            width=20,
        )
        TkToolTip(
            entry_meas_name,
            'The name of the measurement to be saved in the measurement HDF file.',
        )

        lbl_meas_name.grid(row=r, column=1, pady=2, padx=2, sticky='nsw', columnspan=2)
        r += 1
        entry_meas_name.grid(row=r, column=1, pady=2, padx=2, sticky='nsw')
        r += 1

    def _enable_btns(self) -> None:
        """
        Enables buttons depending on if ImageAcquisition, ImageProjection,
        or both are loaded

        """
        # Check ImageAcquisition
        if self.image_acquisition is not None:
            state_im_acqu = 'normal'
        else:
            state_im_acqu = 'disabled'
        # Check ImageProjection
        if self.image_projection is not None:
            state_projection = 'normal'
        else:
            state_projection = 'disabled'
        # Check System
        if self.image_projection is not None and self.image_acquisition is not None:
            state_system = 'normal'
        else:
            state_system = 'disabled'

        # Turn system buttons on/off
        self.btn_gray_levels_cal.config(state=state_system)
        self.btn_run_measurement.config(state=state_system)
        self.btn_load_gray_levels_cal.config(state=state_system)
        self.btn_view_gray_levels_cal.config(state=state_system)

        # Turn projector buttons on/off
        self.btn_show_cal_image.config(state=state_projection)
        self.btn_show_axes.config(state=state_projection)
        self.btn_show_crosshairs.config(state=state_projection)
        self.btn_close_projection.config(state=state_projection)

        # Turn camera only buttons on/off
        self.btn_exposure_cal.config(state=state_im_acqu)
        self.btn_set_exposure.config(state=state_im_acqu)
        self.btn_save_snapshot.config(state=state_im_acqu)
        self.btn_show_snapshot.config(state=state_im_acqu)
        self.btn_live_view.config(state=state_im_acqu)
        self.btn_close_camera.config(state=state_im_acqu)

    def _check_system_loaded(self) -> None:
        """Checks if the system class has been instantiated"""
        if self.system is None:
            messagebox.showerror(
                'Error', 'Both ImageAcquisiton and ImageProjection must both be loaded.'
            )
            return

    def _check_calibration_loaded(self) -> bool:
        """Checks if calibration is loaded. Returns True, if loaded,
        returns False and shows error message if not loaded.
        """
        if self.calibration is None:  # Not loaded
            messagebox.showerror(
                'Error', 'Camera-Projector calibration must be loaded/performed.'
            )
            return False
        else:  # Loaded
            return True

    def _load_system_elements(self) -> None:
        """
        Checks if System object can be instantiated (if both
        ImageAcquisition and ImageProjection classes are loaded)

        """
        if self.image_acquisition is not None and self.image_projection is not None:
            self.system = SystemSofastFringe(
                self.image_projection, self.image_acquisition
            )

    def _save_measurement_data(self, file: str) -> None:
        """Saves last measurement to HDF file"""
        # Check measurement images have been captured
        if (
            self.system.fringe_images_captured is None
            or self.system.mask_images_captured is None
        ):
            raise ValueError('Measurement data has not been captured.')
        elif self.calibration is None:
            raise ValueError('Calibration data has not been processed.')

        # Check save name is valid
        if file == '':
            raise ValueError('Valid file name required to save measurement data.')

        # Get user inputs
        meas_pt = Vxyz(list(map(float, self.var_meas_pt.get().split(','))))
        meas_dist = float(self.var_meas_dist.get())
        meas_name = self.var_meas_name.get()

        # Create measurement object
        measurement = self.system.get_measurements(meas_pt, meas_dist, meas_name)[0]

        # Save measurement
        measurement.save_to_hdf(file)

        # Save calibration data
        self.calibration.save_to_hdf(file)

    def load_image_acquisition(self) -> None:
        """Loads and connects to ImageAcquisition"""
        # Get selected camera object and run
        idx = self.cam_options.index(self.var_cam_select.get())
        self.image_acquisition = self.cam_objects[idx]()

        print('Camera connected.')

        # Enable buttons
        self._enable_btns()

        # Instantiate system class if possible
        self._load_system_elements()

    def load_image_projection(self) -> None:
        """Loads and displays ImageProjection"""
        # Get file name
        file = askopenfilename(
            defaultextension='.h5', filetypes=[("HDF5 File", "*.h5")]
        )

        # Load file and display
        if file != '':
            # Load data
            image_projection_data = ImageProjection.load_from_hdf(file)
            # Create new window
            projector_root = tkinter.Toplevel(self.root)
            # Show window
            self.image_projection = ImageProjection(
                projector_root, image_projection_data
            )

            print(f'ImageProjection loaded:\n    {file}')

            # Enable buttons
            self._enable_btns()

            # Instantiate system class if possible
            self._load_system_elements()

    def show_calibration_image(self) -> None:
        """Shows calibration image"""
        self.image_projection.show_calibration_image()

    def show_crosshairs(self) -> None:
        """Shows crosshairs"""
        self.image_projection.show_crosshairs()

    def show_axes(self) -> None:
        """Shows ImageProjection axes"""
        self.image_projection.show_axes()

    def close_projection_window(self) -> None:
        """Close only projection window"""
        self.image_projection.close()

        # Remove image projection and update buttons
        self.image_projection = None
        self._enable_btns()

    def run_measurement(self) -> None:
        """Runs data collect and saved data."""

        # Get save file name
        file = asksaveasfilename(
            defaultextension='.h5', filetypes=[("HDF5 File", "*.h5")]
        )
        if file == '':
            return

        # Check system object exists
        self._check_system_loaded()

        # Check if calibration file is loaded
        if not self._check_calibration_loaded():
            return

        # Get fringe periods
        periods_x = [4**idx for idx in range(self.var_fringe_periods_x.get())]
        periods_y = [4**idx for idx in range(self.var_fringe_periods_y.get())]
        periods_x[0] -= 0.1
        periods_y[0] -= 0.1

        # Create fringe object
        fringes = Fringes(periods_x, periods_y)

        # Get minimum display value from calibration
        min_disp_value = self.calibration.calculate_min_display_camera_values()[0]

        # Load fringes
        self.system.load_fringes(fringes, min_disp_value)

        # Define what to run after measurement sequence
        def run_next():
            # Save measurement data
            self._save_measurement_data(file)
            # Show crosshairs image
            self.show_crosshairs()
            # Display message
            print(f'Measurement data saved to:\n    {file}')

        # Capture images
        self.system.capture_mask_and_fringe_images(run_next)

    def run_exposure_cal(self) -> None:
        """Runs camera exposure calibration"""
        # If only camera is loaded
        if self.image_acquisition is not None and self.image_projection is None:
            print('Running calibration without displayed white image.')
            self.image_acquisition.calibrate_exposure()
        elif self.image_acquisition is not None and self.image_projection is not None:
            print('Running calibration with displayed white image.')
            run_next = self.show_crosshairs
            self.system.run_camera_exposure_calibration(run_next)
        else:
            messagebox.showerror('Error', 'Camera must be connected.')
            return

    def run_gray_levels_cal(self) -> None:
        """Runs the projector-camera intensity calibration"""
        # Get save file name
        file_default = dt.datetime.now().strftime(
            'projector_camera_response_%Y_%m_%d-%H_%M_%S.h5'
        )
        file = asksaveasfilename(
            initialfile=file_default, filetypes=[("HDF5 File", "*.h5")]
        )

        if file == '':
            print('No file selected.')
            return

        # Check system class exists
        self._check_system_loaded()

        # Get selected calibration object
        idx = self.cal_options.index(self.var_cal_select.get())
        cal_object = self.cal_objects[idx]

        # Capture images
        def _func_0():
            print('Calibrating...')
            self.system.run_display_camera_response_calibration(
                res=10, run_next=self.system.run_next_in_queue
            )

        # Process data
        def _func_1():
            print('Processing calibration data')
            # Get calibration images from System
            calibration_images = self.system.get_calibration_images()[
                0
            ]  # only calibrating one camera
            # Load calibration object
            self.calibration = cal_object.from_data(
                calibration_images, self.system.calibration_display_values
            )
            # Save calibration object
            self.calibration.save_to_hdf(file)
            # Save calibration raw data
            data = [self.system.calibration_display_values, calibration_images]
            datasets = [
                'CalibrationRawData/display_values',
                'CalibrationRawData/images',
            ]
            hdf5_tools.save_hdf5_datasets(data, datasets, file)
            # Show crosshairs
            self.show_crosshairs()
            # Display message
            self.var_gray_lvl_cal_status.set('Calibration data: Loaded/Saved')
            print(f'Calibration complete. Results loaded and saved to\n    {file}')
            # Continue
            self.system.run_next_in_queue()

        self.system.set_queue([_func_0, _func_1])
        self.system.run_next_in_queue()

    def load_gray_levels_cal(self) -> None:
        """Loads saved results of a projector-camera intensity calibration"""
        # Get file name
        file = askopenfilename(
            defaultextension='.h5', filetypes=[("HDF5 File", "*.h5")]
        )

        if file == '':
            return

        # Load file
        cal_type = hdf5_tools.load_hdf5_datasets(
            ['Calibration/calibration_type'], file
        )['calibration_type']

        if cal_type == 'ImageCalibrationGlobal':
            self.calibration = ImageCalibrationGlobal.load_from_hdf(file)
        elif cal_type == 'ImageCalibrationScaling':
            self.calibration = ImageCalibrationScaling.load_from_hdf(file)
        else:
            raise ValueError(f'Selected calibration type, {cal_type}, not supported.')

        # Display message
        self.var_gray_lvl_cal_status.set('Calibration data: Loaded')
        print(f'Calibration type: {cal_type:s}')
        print(f'Calibration file loaded:\n    {file:s}')

    def view_gray_levels_cal(self) -> None:
        """Shows plot of gray levels calibration data"""
        # Check if calibration file is loaded
        if not self._check_calibration_loaded():
            return

        # Plot figure
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(self.calibration.display_values, self.calibration.camera_values)
        ax.set_xlabel('Display Values')
        ax.set_ylabel('Camera Values')
        ax.grid(True)
        ax.set_title('Projector-Camera Calibration Curve')

        plt.show()

    def set_exposure(self) -> None:
        """Sets camera exposure time value to user defined value"""
        # Get current exposure_time value
        cur_exp = self.image_acquisition.exposure_time

        # Get new exposure_time value
        new_exp = simpledialog.askfloat(
            title="Set exposure value",
            prompt=f"Current exposure: {cur_exp:.1f}. New Value:",
        )

        # Set new exposure_time value
        if new_exp is not None:
            self.image_acquisition.exposure_time = new_exp

    def save_snapshot(self) -> None:
        """Save current snapshot from camera"""
        # Show image
        frame = self._get_frame()

        # Get save file name
        file = asksaveasfilename(
            defaultextension='.png', filetypes=[("All tpyes", "*.*")]
        )

        # Save image
        if file != '':
            imageio.imsave(file, frame.astype(np.uint8))

    def show_snapshot(self) -> None:
        """
        Captures frame and displays image and image histogram.

        Returns
        -------
        frame : ndarray
            2D image array.

        """
        # Get frame
        frame = self._get_frame()

        # Highlight saturation
        frame_rgb = highlight_saturation(frame, self.image_acquisition.max_value)

        # Create figure
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Plot image
        self._plot_image(
            ax1, frame_rgb, 'Current camera view', 'X (pixels)', 'Y (pixels)'
        )

        # Plot histogram
        self._plot_hist(ax2, frame)

        # Show entire figure
        plt.show()

    def live_view(self) -> None:
        """Shows live stream from the camera."""
        LiveView(self.image_acquisition)

    def close_image_acquisition(self) -> None:
        """Closes connection to camera"""
        # Close camera
        self.image_acquisition.close()

        # Clear objects
        self.image_acquisition = None
        self.system = None
        self._enable_btns()

    def close(self) -> None:
        """
        Closes all windows

        """
        # Close image projection
        if self.image_projection is not None:
            self.image_projection.close()

        # Close image acquisition
        if self.image_acquisition is not None:
            self.image_acquisition.close()

        # Close Sofast window
        self.root.destroy()


if __name__ == '__main__':
    SofastGUI()
