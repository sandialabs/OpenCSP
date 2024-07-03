"""GUI to control SOFAST image projection, image acquisition,
and data capture. Can capture datasets and save to HDF format.
"""

import datetime as dt
import functools
import os
import tkinter
from tkinter import messagebox, simpledialog
from tkinter.filedialog import askopenfilename, asksaveasfilename

import imageio
import matplotlib.pyplot as plt
import numpy as np

from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.app.sofast.lib.ImageCalibrationAbstract import ImageCalibrationAbstract
import opencsp.app.sofast.lib.sofast_common_functions as scf
import opencsp.app.sofast.lib.SystemSofastFringe as ssf
from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract
from opencsp.common.lib.camera.image_processing import highlight_saturation
from opencsp.common.lib.camera.LiveView import LiveView
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection, ImageProjectionData
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.tk_tools as tkt
import opencsp.common.lib.tool.time_date_tools as tdt


class SofastGUI:
    """Class that contains SOFAST GUI controls and methods"""

    def __init__(self) -> 'SofastGUI':
        """
        Instantiates GUI in new window
        """
        # Placeholder for system instances
        self.sys_fringe: ssf.SystemSofastFringe = None

        # Create tkinter object
        self.root = tkt.window()

        # Set title
        self.root.title('SOFAST')

        # Set size of GUI
        self.root.geometry('600x680+200+100')

        # Add all buttons/widgets to window
        self._create_layout()

        # Disable buttons
        self._enable_btns()

        # Run window infinitely
        self.root.mainloop()

    def _load_system_fringe_elements(self) -> bool:
        """Loads the system instance, as appropriate.

        Checks if System object can be instantiated (if both ImageAcquisition and ImageProjection classes are loaded)
        """
        camera = ImageAcquisitionAbstract.instance()
        projector = ImageProjection.instance()
        if camera is not None and projector is not None:
            # Check that we don't already have a fringe instance
            if self.sys_fringe is not None:
                return True
            # Create the new instance
            self.sys_fringe = ssf.SystemSofastFringe()
            # Update the state of the GUI
            self._enable_btns()
            return True
        return False

    def _get_frame(self) -> np.ndarray | None:
        """
        Captures frame from camera

        Returns
        -------
        frame : ndarray
            2D image from camera.

        """
        # Check that we have a camera
        if not self._check_camera_loaded('_get_frame'):
            return

        frame = ImageAcquisitionAbstract.instance().get_frame()
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

    def _plot_image(self, ax: plt.Axes, image: np.ndarray, title: str = '', xlabel: str = '', ylabel: str = '') -> None:
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

    @functools.cached_property
    def cam_options(self):
        """Returns all camera (ImageAcquisition) options that are compatible with Sofast"""
        all_cam_options = ImageAcquisitionAbstract.cam_options()

        ret: dict[str, type[ImageAcquisitionAbstract]] = {}
        for description in all_cam_options:
            if "file" not in description.lower():
                ret[description] = all_cam_options[description]

        return ret

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
        label_frame_settings.grid(row=0, column=1, rowspan=3, sticky='nesw', padx=5, pady=5)

        # =============== First Column - Load components ===============
        r = 0
        # Connect camera button
        self.btn_load_image_acquisition = tkinter.Button(
            label_frame_load_system, text='Connect Camera', command=self.load_image_acquisition
        )
        self.btn_load_image_acquisition.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(
            self.btn_load_image_acquisition,
            'Select the type of camera in the dropdown menu to the right. Click to connect to the camera.',
        )
        r += 1

        # Load physical layout
        self.btn_load_image_projection = tkinter.Button(
            label_frame_load_system, text='Load ImageProjection', command=self.load_image_projection
        )
        self.btn_load_image_projection.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(self.btn_load_image_projection, 'Select an ImageProjection HDF file to load and display.')
        r += 1

        # =============== First Column - Projection controls ===============
        r = 0
        self.btn_show_cal_image = tkinter.Button(
            label_frame_projector, text='Show Cal Fiducial Image', command=self.show_calibration_fiducial_image
        )
        self.btn_show_cal_image.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(self.btn_show_cal_image, 'Shows calibration fiducial image on projection window.')
        r += 1

        self.btn_show_cal_image = tkinter.Button(
            label_frame_projector, text='Show Cal Marker Image', command=self.show_calibration_marker_image
        )
        self.btn_show_cal_image.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(self.btn_show_cal_image, 'Shows calibration marker image on projection window.')
        r += 1

        self.btn_show_axes = tkinter.Button(label_frame_projector, text='Show Screen Axes', command=self.show_axes)
        self.btn_show_axes.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(self.btn_show_axes, 'Shows screen axes on projection window.')
        r += 1

        self.btn_show_crosshairs = tkinter.Button(
            label_frame_projector, text='Show Crosshairs', command=self.show_crosshairs
        )
        self.btn_show_crosshairs.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(self.btn_show_crosshairs, 'Shows crosshairs on projection window.')
        r += 1

        self.btn_close_projection = tkinter.Button(
            label_frame_projector, text='Close Display Window', command=self.close_projection_window
        )
        self.btn_close_projection.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(self.btn_close_projection, 'Close only projection window.')
        r += 1

        # =============== First Column - Camera controls ===============
        r = 0
        # Perform exposure calibration
        self.btn_exposure_cal = tkinter.Button(
            label_frame_camera, text='Calibrate Exposure Time', command=self.run_exposure_cal
        )
        self.btn_exposure_cal.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(self.btn_exposure_cal, 'Automatically performs camera exposure calibration.')
        r += 1

        # Set camera exposure
        self.btn_set_exposure = tkinter.Button(label_frame_camera, text='Set Exposure Time', command=self.set_exposure)
        self.btn_set_exposure.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(
            self.btn_set_exposure, 'Set the camera exposure time value. The current value is displayed when clicked.'
        )
        r += 1

        # Show snapshot button
        self.btn_show_snapshot = tkinter.Button(label_frame_camera, text='Show Snapshot', command=self.show_snapshot)
        self.btn_show_snapshot.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(self.btn_show_snapshot, 'Shows a camera image and pixel brightness histogram.')
        r += 1

        # Save snapshot button
        self.btn_save_snapshot = tkinter.Button(label_frame_camera, text='Save Snapshot', command=self.save_snapshot)
        self.btn_save_snapshot.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(self.btn_save_snapshot, 'Saves image from camera as a PNG.')
        r += 1

        # Live view button
        self.btn_live_view = tkinter.Button(label_frame_camera, text='Live View', command=self.live_view)
        self.btn_live_view.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(self.btn_live_view, 'Shows live view from connected camera.')
        r += 1

        # Perform exposure calibration
        self.btn_close_camera = tkinter.Button(
            label_frame_camera, text='Close Camera', command=self.close_image_acquisition
        )
        self.btn_close_camera.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(self.btn_close_camera, 'Closes connection to camera.')
        r += 1

        # =============== First Column - System run controls ===============
        r = 0
        # Run sofast capture
        self.btn_run_measurement = tkinter.Button(
            label_frame_run, text='Run Data Capture', command=self.run_measurement
        )
        self.btn_run_measurement.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(self.btn_run_measurement, 'Runs SOFAST data capture. Mask then fringes are captured.')
        r += 1

        # Perform projector-camera brightness calibration
        self.btn_gray_levels_cal = tkinter.Button(
            label_frame_run, text='Run Response Calibration', command=self.run_gray_levels_cal
        )
        self.btn_gray_levels_cal.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(
            self.btn_gray_levels_cal,
            'Performs Projector-Camera Brightness Calibration sequence. Must select a destination HDF file first.',
        )
        r += 1

        # Load projector-camera brightness calibration
        self.btn_load_gray_levels_cal = tkinter.Button(
            label_frame_run, text='Load Response Calibration', command=self.load_gray_levels_cal
        )
        self.btn_load_gray_levels_cal.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(
            self.btn_load_gray_levels_cal,
            'Loads a previously saved Projector-Camera Brightness Calibration sequence data file.',
        )
        r += 1

        # View projector-camera brightness calibration
        self.btn_view_gray_levels_cal = tkinter.Button(
            label_frame_run, text='View Response Calibration', command=self.view_gray_levels_cal
        )
        self.btn_view_gray_levels_cal.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(
            self.btn_view_gray_levels_cal, 'Views current Projector-Camera Brightness Calibration sequence data file.'
        )
        r += 1

        # =============== First Column - Close button ===============
        # Close window button
        self.btn_close = tkinter.Button(self.root, text='Close All', command=self.close)
        self.btn_close.grid(row=4, column=0, pady=2, padx=2, sticky='nesw')
        tkt.TkToolTip(self.btn_close, 'Closes all windows.')

        # =============== Second Column - Settings ===============
        r = 0
        # Camera type dropdown
        self.var_cam_select = tkinter.StringVar(value=list(self.cam_options.keys())[0])
        lbl_camera_type = tkinter.Label(label_frame_settings, text='Select camera:', font=('calibre', 10, 'bold'))
        cam_options = list(self.cam_options.keys())
        drop_camera_type = tkinter.OptionMenu(
            label_frame_settings, self.var_cam_select, cam_options[0], *cam_options[1:]
        )
        tkt.TkToolTip(drop_camera_type, 'Select type of camera object to load.')

        lbl_camera_type.grid(row=r, column=1, pady=2, padx=2, sticky='nse')
        drop_camera_type.grid(row=r, column=2, pady=2, padx=2, sticky='nsw')
        r += 1

        # Calibration type dropdown
        cal_options = ImageCalibrationAbstract.get_cal_options()
        self.var_cal_select = tkinter.StringVar(value=list(cal_options.keys())[0])
        lbl_cal_type = tkinter.Label(
            label_frame_settings, text='Select calibration method:', font=('calibre', 10, 'bold')
        )
        drop_cal_type = tkinter.OptionMenu(label_frame_settings, self.var_cal_select, *list(cal_options.keys()))
        tkt.TkToolTip(drop_cal_type, 'Select type of Projector-Camera Brightness Calibration process to use.')

        lbl_cal_type.grid(row=r, column=1, pady=2, padx=2, sticky='nse')
        drop_cal_type.grid(row=r, column=2, pady=2, padx=2, sticky='nsw')
        r += 1

        # Fringe periods input box
        self.var_fringe_periods_x = tkinter.IntVar(value=4)
        self.var_fringe_periods_y = tkinter.IntVar(value=4)
        lbl_fringe_x = tkinter.Label(label_frame_settings, text='Fringe X periods:', font=('calibre', 10, 'bold'))
        lbl_fringe_y = tkinter.Label(label_frame_settings, text='Fringe Y periods:', font=('calibre', 10, 'bold'))
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
        tkt.TkToolTip(
            entry_fringe_x,
            'Number of fringe X periods to show during a data capture. Each fringe period will always be phase shifted four times.',
        )
        tkt.TkToolTip(
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
        self.var_gray_lvl_cal_status = tkinter.StringVar(value='Calibration data: No Data')
        lbl_gray_lvl_cal_status = tkinter.Label(
            label_frame_settings, textvariable=self.var_gray_lvl_cal_status, font=('calibre', 10, 'bold')
        )

        lbl_gray_lvl_cal_status.grid(row=r, column=1, pady=2, padx=2, sticky='nsw', columnspan=2)
        r += 1

        # Measure point input
        self.var_meas_pt = tkinter.StringVar(value='0.0, 0.0, 0.0')
        lbl_meas_pt = tkinter.Label(label_frame_settings, text='Measure Point XYZ', font=('calibre', 10, 'bold'))
        entry_meas_pt = tkinter.Entry(
            label_frame_settings, textvariable=self.var_meas_pt, font=('calibre', 10, 'normal'), width=20
        )
        tkt.TkToolTip(
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
            label_frame_settings, text='Measured mirror-screen distance', font=('calibre', 10, 'bold')
        )
        entry_meas_dist = tkinter.Entry(
            label_frame_settings, textvariable=self.var_meas_dist, font=('calibre', 10, 'normal'), width=20
        )
        tkt.TkToolTip(
            entry_meas_dist,
            'The distance in meters from the display crosshairs to the "Measure Point XYZ" location on the mirror.',
        )

        lbl_meas_dist.grid(row=r, column=1, pady=2, padx=2, sticky='nsw', columnspan=2)
        r += 1
        entry_meas_dist.grid(row=r, column=1, pady=2, padx=2, sticky='nsw')
        r += 1

        # Measurement name
        self.var_meas_name = tkinter.StringVar(value='')
        lbl_meas_name = tkinter.Label(label_frame_settings, text='Measurement name', font=('calibre', 10, 'bold'))
        entry_meas_name = tkinter.Entry(
            label_frame_settings, textvariable=self.var_meas_name, font=('calibre', 10, 'normal'), width=20
        )
        tkt.TkToolTip(entry_meas_name, 'The name of the measurement to be saved in the measurement HDF file.')

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
        if ImageAcquisitionAbstract.instance() is not None:
            state_im_acqu = 'normal'
        else:
            state_im_acqu = 'disabled'
        # Check ImageProjection
        if ImageProjection.instance() is not None:
            state_projection = 'normal'
        else:
            state_projection = 'disabled'
        # Check System
        if ImageProjection.instance() is not None and ImageAcquisitionAbstract.instance() is not None:
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

    def _check_system_fringe_loaded(self, method_name: str) -> bool:
        """Checks if the system class has been instantiated, and that the camera and projector instances are still available"""
        try:
            return scf.check_system_fringe_loaded(self.sys_fringe, method_name)
        except Exception as ex:
            messagebox.showerror('Error', repr(ex))
            return False

    def _check_calibration_loaded(self, method_name: str) -> bool:
        """Checks if calibration is loaded. Returns True if loaded."""
        try:
            return scf.check_calibration_loaded(self.sys_fringe, method_name)
        except Exception as ex:
            messagebox.showerror('Error', repr(ex))
            return False

    def _check_camera_loaded(self, method_name: str) -> bool:
        try:
            return scf.check_camera_loaded(method_name)
        except Exception as ex:
            messagebox.showerror('Error', repr(ex))
            return False

    def _check_projector_loaded(self, method_name: str) -> bool:
        try:
            return scf.check_projector_loaded(method_name)
        except Exception as ex:
            messagebox.showerror('Error', repr(ex))
            return False

    def _save_measurement_data(self, file: str) -> None:
        """Saves last measurement to HDF file"""
        if not self._check_system_fringe_loaded('_save_measurement_data'):
            return

        # Check measurement images have been captured
        sys = self.sys_fringe
        if sys.calibration is None:
            raise ValueError('Image Calibration data has not been processed.')

        # Check save name is valid
        if file == '':
            raise ValueError('Valid file name required to save measurement data.')

        # Get user inputs
        meas_pt = Vxyz(list(map(float, self.var_meas_pt.get().split(','))))
        meas_dist = float(self.var_meas_dist.get())
        meas_name = self.var_meas_name.get()

        # Create measurement object
        measurement = sys.get_measurements(meas_pt, meas_dist, meas_name)[0]

        # Save measurement
        measurement.save_to_hdf(file)

        # Save calibration data
        sys.calibration.save_to_hdf(file)

    def load_image_acquisition(self) -> None:
        """Loads and connects to ImageAcquisition"""
        # Get selected camera class
        cam_description = self.var_cam_select.get()
        # Close the previous camera instance
        with et.ignored(Exception):
            ImageAcquisitionAbstract.instance().close()
        # Create instance of the camera
        ia = self.cam_options[cam_description]()
        # Try to load the system instance
        self._load_system_fringe_elements()
        # Register callbacks
        ImageAcquisitionAbstract.instance().on_close.append(self._camera_closed)
        # Update the state of the GUI
        self._enable_btns()

        lt.info('Camera connected.')

    def load_image_projection(self) -> None:
        """Loads and displays ImageProjection"""
        # Get file name
        file = askopenfilename(defaultextension='.h5', filetypes=[("HDF5 File", "*.h5")])

        # Load file and display
        if file != '':
            # Close the previous projector instance
            with et.ignored(Exception):
                ImageProjection.instance().close()
            # Create new window
            projector_root = tkt.window(self.root, TopLevel=True)
            # Load data
            image_projection_data = ImageProjectionData.load_from_hdf(file)
            # Show window
            ImageProjection(projector_root, image_projection_data)
            # Build system instance
            self._load_system_fringe_elements()
            # Register callbacks
            ImageProjection.instance().on_close.append(self._projector_closed)
            # Update the state of the GUI
            self._enable_btns()

            lt.info(f'ImageProjection loaded:\n    {file}')

    def show_calibration_fiducial_image(self) -> None:
        """Shows calibration image with fiducials"""
        # Check that we have a projector
        if not self._check_projector_loaded('show_calibration_fiducial_image'):
            return

        ImageProjection.instance().show_calibration_fiducial_image()

    def show_calibration_marker_image(self) -> None:
        """Shows calibration image with Aruco markers"""
        # Check that we have a projector
        if not self._check_projector_loaded('show_calibration_marker_image'):
            return

        ImageProjection.instance().show_calibration_marker_image()

    def show_crosshairs(self) -> None:
        """Shows crosshairs"""

        # Check that we have a projector
        if not self._check_projector_loaded('show_crosshairs'):
            return

        ImageProjection.instance().show_crosshairs()

    def show_axes(self) -> None:
        """Shows ImageProjection axes"""

        # Check that we have a projector
        if not self._check_projector_loaded('show_axes'):
            return

        ImageProjection.instance().show_axes()

    def close_projection_window(self) -> None:
        """Close only projection window"""

        # Check that we have a projector
        if not self._check_projector_loaded('close_projection_window'):
            return

        ImageProjection.instance().close()

    def run_measurement(self) -> None:
        """Runs data collect and saved data."""

        # Get save file name
        file = asksaveasfilename(defaultextension='.h5', filetypes=[("HDF5 File", "*.h5")])
        if file == '':
            return

        # Check system object exists
        if not self._check_system_fringe_loaded('run_measurement'):
            return

        # Check if calibration file is loaded
        if not self._check_calibration_loaded('run_measurment'):
            return

        # Get fringe object
        fringe_periods_x = self.var_fringe_periods_x.get()
        fringe_periods_y = self.var_fringe_periods_y.get()
        fringes = Fringes.from_num_periods(fringe_periods_x, fringe_periods_y)
        self.sys_fringe.set_fringes(fringes)

        # Define what to run after measurement sequence
        def run_next():
            # Save measurement data
            self._save_measurement_data(file)
            # Show crosshairs image
            self.show_crosshairs()
            # Display message
            lt.info(f'Measurement data saved to:\n    {file}')

        # Capture images
        self.sys_fringe.run_measurement(run_next)

    def run_exposure_cal(self) -> None:
        """Runs camera exposure calibration"""
        try:
            scf.run_exposure_cal()
        except Exception as ex:
            messagebox.showerror('Error', repr(ex))

    def run_gray_levels_cal(self) -> None:
        """Runs the projector-camera intensity calibration"""
        # Get save file name
        file_default = dt.datetime.now().strftime('projector_camera_response_%Y_%m_%d-%H_%M_%S.h5')
        file = asksaveasfilename(initialfile=file_default, filetypes=[("HDF5 File", "*.h5")])

        if file == '':
            lt.info('No file selected.')
            return

        # Check system class exists
        if not self._check_system_fringe_loaded('run_gray_levels_cal'):
            return

        # Get selected calibration type
        cal_type = ImageCalibrationAbstract.get_cal_options()[self.var_cal_select.get()]

        # Let us know when each phase is starting/has finished
        def on_capturing():
            lt.info('Calibrating...')

        def on_processing():
            lt.info('Processing calibration data')

        def on_processed():
            # Show crosshairs
            ImageProjection.instance().show_crosshairs()

            # Enable buttons
            self._enable_btns()

            # Notify user
            self.var_gray_lvl_cal_status.set('Calibration data: Loaded/Saved')
            lt.info(f'Calibration complete. Results loaded and saved to\n    {file}')

        # Run calibration
        self.sys_fringe.run_gray_levels_cal(
            calibration_class=cal_type,
            calibration_hdf5_path_name_ext=file,
            on_capturing=on_capturing,
            on_processing=on_processing,
            on_processed=on_processed,
        )

    def load_gray_levels_cal(self) -> None:
        """Loads saved results of a projector-camera intensity calibration"""
        # Check that the system instance exists
        if not self._check_system_fringe_loaded('load_gray_levels_cal'):
            return

        # Get file name
        file = askopenfilename(defaultextension='.h5', filetypes=[("HDF5 File", "*.h5")])

        if file == '':
            return

        # Load the calibration
        try:
            self.sys_fringe.set_calibration(ImageCalibrationAbstract.load_from_hdf_guess_type(file))
        except Exception as ex:
            messagebox.showerror('Error', repr(ex))

        # Enable buttons
        self._enable_btns()

        # Display message
        cal_type = self.sys_fringe.calibration.get_calibration_name()
        self.var_gray_lvl_cal_status.set('Calibration data: Loaded')
        lt.info(f'Calibration type: {cal_type:s}')
        lt.info(f'Calibration file loaded:\n    {file:s}')

    def view_gray_levels_cal(self) -> None:
        """Shows plot of gray levels calibration data"""
        # Check if calibration file is loaded
        if not self._check_calibration_loaded('view_gray_levels_cal'):
            return

        # Plot figure
        self.sys_fringe.calibration.plot_gray_levels()

        plt.show()

    def set_exposure(self) -> None:
        """Sets camera exposure time value to user defined value"""
        # Get current exposure_time value
        try:
            cur_exp = scf.get_exposure()
        except RuntimeError:
            cur_exp = None

        # Get new exposure_time value
        new_exp = simpledialog.askfloat(
            title="Set exposure value", prompt=f"Current exposure: {cur_exp:.1f}. New Value:"
        )

        # Set new exposure_time value
        if new_exp is not None:
            scf.set_exposure(new_exp)

    def save_snapshot(self) -> None:
        """Save current snapshot from camera"""
        # Show image
        frame = self._get_frame()
        if frame is None:
            return

        # Get save file name
        file = asksaveasfilename(defaultextension='.png', filetypes=[("All tpyes", "*.*")])

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
        if frame is None:
            return

        # Highlight saturation
        frame_rgb = highlight_saturation(frame, ImageAcquisitionAbstract.instance().max_value)

        # Create figure
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Plot image
        self._plot_image(ax1, frame_rgb, 'Current camera view', 'X (pixels)', 'Y (pixels)')

        # Plot histogram
        self._plot_hist(ax2, frame)

        # Show entire figure
        plt.show()

    def live_view(self) -> None:
        """Shows live stream from the camera."""
        LiveView()

    def _camera_closed(self, image_acquisition: ImageAcquisitionAbstract):
        # Release the system instance
        self.sys_fringe = None
        # Update the state of the GUI
        self._enable_btns()

    def _projector_closed(self, image_projection: ImageProjection):
        self._camera_closed(None)

    def close_image_acquisition(self) -> None:
        """
        Close the single Image Acquisition object if open.
        """
        # Close the camera
        with et.ignored(Exception):
            ImageAcquisitionAbstract.instance().close()
        # Update associated values
        # self._camera_closed() # should happen automatically via callbacks

    def close(self) -> None:
        """
        Closes all windows

        """
        with et.ignored(Exception):
            self.close_image_acquisition()

        with et.ignored(Exception):
            self.sys_fringe.close_all()

        # Close Sofast window
        with et.ignored(Exception):
            self.root.destroy()


if __name__ == '__main__':
    # start the logger
    log_name_ext = "SOFAST_GUI_" + tdt.current_date_time_string_forfile() + ".log"
    log_path_name_ext = os.path.join(orp.opencsp_temporary_dir(), log_name_ext)
    lt.logger(log_path_name_ext)

    # start the gui
    SofastGUI()
