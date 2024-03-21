"""
GUI used to load previously captured calibration images and calibrate a
machine vision camera.
"""

import os
import tkinter
from tkinter.filedialog import askopenfilename, asksaveasfilename
from warnings import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

import opencsp.app.camera_calibration.lib.calibration_camera as cc
import opencsp.app.camera_calibration.lib.image_processing as ip
from opencsp.app.camera_calibration.lib.ViewAnnotatedImages import ViewAnnotatedImages
import opencsp.app.sofast.lib.spatial_processing as sp
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz


class CalibrationGUI:
    def __init__(self):
        """
        GUI for calibrating machine vision Camera

        """
        # Create tkinter object
        self.root = tkinter.Tk()

        # Set title
        self.root.title('Camera Calibration')

        # Set size of GUI
        self.root.geometry('550x480+200+100')

        # Add all buttons/widgets to window
        self.create_layout()

        # Set flags
        self.files_slected = False
        self.images_loaded = False
        self.camera_calibrated = False

        # Initialize variables
        self.files: list[str]
        self.images: list[np.ndarray]
        self.used_file_names: list[str]
        self.p_object: list[Vxyz]
        self.p_image: list[Vxy]
        self.img_size_xy: tuple[int, int]
        self.camera: Camera
        self.r_cam_object: list[Rotation]
        self.v_cam_object_cam: list[Vxyz]
        self.avg_reproj_error: list[float]
        self.reproj_errors: list[float]

        # Format buttons
        self.enable_btns()

        # Run window infinitely
        self.root.mainloop()

    def create_layout(self):
        """
        Creates GUI widgets

        """
        # BUTTONS / ENTRIES
        r = 0

        # Name of camera input
        self.var_cam_name = tkinter.StringVar(value='Camera')
        lbl_cam_name = tkinter.Label(self.root, text='Camera Name:', font=('calibre', 10, 'bold'))
        entry_cam_name = tkinter.Entry(
            self.root, textvariable=self.var_cam_name, font=('calibre', 10, 'normal'), width=40
        )

        lbl_cam_name.grid(row=r, column=0, pady=2, padx=2, sticky='nsw')
        entry_cam_name.grid(row=r, column=1, pady=2, padx=2, sticky='nsw')
        r += 1

        # Number of points input
        self.var_pts_x = tkinter.IntVar(value=18)
        self.var_pts_y = tkinter.IntVar(value=23)
        lbl_pts_x = tkinter.Label(self.root, text='Number of grid x points:', font=('calibre', 10, 'bold'))
        lbl_pts_y = tkinter.Label(self.root, text='Number of grid y points:', font=('calibre', 10, 'bold'))
        entry_pts_x = tkinter.Entry(self.root, textvariable=self.var_pts_x, font=('calibre', 10, 'normal'), width=10)
        entry_pts_y = tkinter.Entry(self.root, textvariable=self.var_pts_y, font=('calibre', 10, 'normal'), width=10)

        lbl_pts_x.grid(row=r, column=0, pady=2, padx=2, sticky='nsw')
        entry_pts_x.grid(row=r, column=1, pady=2, padx=2, sticky='nsw')
        r += 1
        lbl_pts_y.grid(row=r, column=0, pady=2, padx=2, sticky='nsw')
        entry_pts_y.grid(row=r, column=1, pady=2, padx=2, sticky='nsw')
        r += 1

        # Select images button
        self.btn_select_ims = tkinter.Button(self.root, text='Select Images', command=self.select_images)
        self.btn_select_ims.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        r += 1

        # Find corners button
        self.btn_find_corns = tkinter.Button(self.root, text='Find Corners', command=self.find_corners)
        self.btn_find_corns.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        r += 1

        # View annotated images button
        self.btn_view_corns = tkinter.Button(self.root, text='View Found Corners', command=self.view_found_corners)
        self.btn_view_corns.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        r += 1

        # Calibrate button
        self.btn_calibrate = tkinter.Button(self.root, text='Calibrate Camera', command=self.calibrate_camera)
        self.btn_calibrate.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        r += 1

        # Show reprojection error button
        self.btn_vis_reproj_error = tkinter.Button(
            self.root, text='Show reprojection error', command=self.show_reproj_error
        )
        self.btn_vis_reproj_error.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        r += 1

        # Visualize distortion
        self.btn_vis_dist = tkinter.Button(self.root, text='Visualize distortion', command=self.visualize_dist)
        self.btn_vis_dist.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        r += 1

        # Save camera button
        self.btn_save = tkinter.Button(self.root, text='Save Camera', command=self.save_camera)
        self.btn_save.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        r += 1

        # Close button
        self.btn_close = tkinter.Button(self.root, text='Close', command=self.close)
        self.btn_close.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')
        r += 1

        # Reprojection error labels
        lbl = tkinter.Label(text='Reprojection Error', borderwidth=1, relief='solid', font=('calibre', 10, 'bold'))
        lbl.grid(row=r, column=0, sticky='nsew', pady=(20, 0))
        lbl = tkinter.Label(text='Image Name', width=40, borderwidth=1, relief='solid', font=('calibre', 10, 'bold'))
        lbl.grid(row=r, column=1, sticky='nsw', pady=(20, 0))
        r += 1

        self.var_reproj_name = []
        self.var_reproj_val = []
        for _ in range(5):
            var_val = tkinter.StringVar(value='')
            var_name = tkinter.StringVar(value='')

            self.var_reproj_val.append(var_val)
            self.var_reproj_name.append(var_name)

            lbl_val = tkinter.Label(textvariable=var_val, borderwidth=1, relief="solid", font=('calibre', 10, 'normal'))
            lbl_name = tkinter.Label(
                textvariable=var_name, borderwidth=1, width=40, relief="solid", font=('calibre', 10, 'normal')
            )

            lbl_val.grid(row=r, column=0, sticky='nsew')
            lbl_name.grid(row=r, column=1, sticky='nsw')

            r += 1

        # LABELS
        r = 3

        # Selected files label
        self.lbl_num_files = tkinter.Label(self.root, font=('calibre', 10, 'bold'))
        self.lbl_num_files.grid(row=r, column=1, pady=2, padx=2, sticky='nsw')
        r += 1

        # Corners found label
        self.lbl_corns_found = tkinter.Label(self.root, font=('calibre', 10, 'bold'))
        self.lbl_corns_found.grid(row=r, column=1, pady=2, padx=2, sticky='nsw')
        r += 1

        # Camera calibrated label
        r += 1
        self.lbl_cam_calibrated = tkinter.Label(self.root, font=('calibre', 10, 'bold'))
        self.lbl_cam_calibrated.grid(row=r, column=1, pady=2, padx=2, sticky='nsw')
        r += 1

    def select_images(self):
        # Asks user to select files
        filetypes = [
            ('all files', '.*'),
            ('PNG files', '*.png'),
            ('JPG files', ('*.jpg', '*.jpeg')),
            ('TIF files', ('*.tif', '*.tiff')),
            ('GIF files', '*.gif'),
        ]
        files = askopenfilename(filetypes=filetypes, title="Select image files", multiple=True)

        if len(files) != '':
            # Save files
            self.files = files

            # Update flags
            self.files_slected = True
            self.images_loaded = False
            self.camera_calibrated = False

            # Clear data
            self.images = []
            self.used_file_names = []
            self.p_object = []
            self.p_image = []
            self.img_size_xy = []
            self.camera = None
            self.r_cam_object = []
            self.v_cam_object_cam = []
            self.avg_reproj_error = []
            self.reproj_errors = []

            # Format buttons
            self.enable_btns()

    def get_npts(self):
        return (self.var_pts_x.get(), self.var_pts_y.get())

    def find_corners(self):
        # Get number checkerboard points
        npts = self.get_npts()

        # Find corners
        self.p_object = []
        self.p_image = []
        self.images = []
        self.used_file_names = []

        for file in self.files:
            # Get file name
            file_name = os.path.basename(file)

            # Update progress
            print('Processing:', file_name, flush=True)

            # Load images
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

            # Find checkerboard corners
            p_object, p_image = ip.find_checkerboard_corners(npts, img)
            if p_object is None or p_image is None:
                warn(f'Could not find corners in image: {file_name:s}.', stacklevel=2)
                continue

            # Save image, filename, and found corners
            self.images.append(img)
            self.used_file_names.append(file_name)

            self.p_object.append(p_object)
            self.p_image.append(p_image)

        # Update flags
        self.images_loaded = True
        self.camera_calibrated = False

        # Save image size
        self.img_size_xy = np.flip(self.images[0].shape)

        # Clear any calibrated camera
        self.camera = None
        self.r_cam_object = []
        self.v_cam_object_cam = []
        self.avg_reproj_error = []
        self.reproj_errors = []

        # Format buttons
        self.enable_btns()

    def view_found_corners(self):
        # Create new window
        root_corns = tkinter.Toplevel(self.root)

        # Get number checkerboard points
        npts = self.get_npts()

        # Annotate images
        ims = []
        for idx, image in enumerate(self.images):
            # Create RGB image
            im3 = np.concatenate([image[..., np.newaxis]] * 3, axis=2)
            # Annotate image
            ip.annotate_found_corners(npts, im3, self.p_image[idx])
            # Save image
            ims.append(im3)

        # View corners
        ViewAnnotatedImages(root_corns, ims, self.used_file_names)

    def calibrate_camera(self):
        # Get camera name
        cam_name = self.var_cam_name.get()

        # Calibrate camera
        (self.camera, self.r_cam_object, self.v_cam_object_cam, self.avg_reproj_error) = cc.calibrate_camera(
            self.p_object, self.p_image, self.img_size_xy, cam_name
        )

        # Calculate reprojection error for each image
        self.reproj_errors = []
        for R_cam, V_cam, P_object, P_image in zip(
            self.r_cam_object, self.v_cam_object_cam, self.p_object, self.p_image
        ):
            error = sp.reprojection_error(self.camera, P_object, P_image, R_cam, V_cam)  # RMS pixels
            self.reproj_errors.append(error)  # RMS pixels

        # Find five images with highest reprojection errors
        idxs = np.flip(np.argsort(self.reproj_errors))[: len(self.var_reproj_name)]
        for idx, name, val in zip(idxs, self.var_reproj_name, self.var_reproj_val):
            name.set(self.used_file_names[idx])
            val.set(f'{self.reproj_errors[idx]:.2f}')

        # Update flags
        self.camera_calibrated = True

        # Format buttons
        self.enable_btns()

    def show_reproj_error(self):
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(self.reproj_errors, 'o-')
        ax.set_xlim(-0.5, len(self.reproj_errors) + 0.5)
        ax.xaxis.set_ticks(np.arange(len(self.reproj_errors)))
        ax.grid()
        ax.set_ylabel('Reprojection Error (Pixels RMS)')
        ax.set_xlabel('Image Index Number')
        plt.show(block=False)

    def visualize_dist(self):
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)

        cc.view_distortion(self.camera, ax1, ax2, ax3)

        plt.show(block=False)

    def save_camera(self):
        file = asksaveasfilename(defaultextension='.h5', filetypes=[("HDF5 File", "*.h5")])

        if file != '':
            self.camera.save_to_hdf(file)

    def enable_btns(self):
        if not self.files_slected:
            # No files selected
            self.btn_find_corns.config(state="disabled")
            self.btn_view_corns.config(state="disabled")
            self.btn_calibrate.config(state="disabled")
            self.btn_vis_reproj_error.config(state="disabled")
            self.btn_vis_dist.config(state="disabled")
            self.btn_save.config(state="disabled")

            self.lbl_num_files.config(text='0 files')
            self.lbl_corns_found.config(text='Corners not found')
            self.lbl_cam_calibrated.config(text='Not calibrated')

            self.clear_reproj_errors()
        elif not self.images_loaded:
            # Files selected, but not not loaded/processed
            self.btn_find_corns.config(state="normal")
            self.btn_view_corns.config(state="disabled")
            self.btn_calibrate.config(state="disabled")
            self.btn_vis_reproj_error.config(state="disabled")
            self.btn_vis_dist.config(state="disabled")
            self.btn_save.config(state="disabled")

            self.lbl_num_files.config(text=f'{len(self.files):d} files')
            self.lbl_corns_found.config(text='Corners not found')
            self.lbl_cam_calibrated.config(text='Not calibrated')

            self.clear_reproj_errors()
        elif self.camera is None:
            # Files selected and processed, but camera not calibrated
            self.btn_find_corns.config(state="normal")
            self.btn_view_corns.config(state="normal")
            self.btn_calibrate.config(state="normal")
            self.btn_vis_reproj_error.config(state="disabled")
            self.btn_vis_dist.config(state="disabled")
            self.btn_save.config(state="disabled")

            self.lbl_num_files.config(text=f'{len(self.files):d} files')
            self.lbl_corns_found.config(text='All corners found')
            self.lbl_cam_calibrated.config(text='Not calibrated')

            self.clear_reproj_errors()
        else:
            # Camera calibrated
            self.btn_find_corns.config(state="normal")
            self.btn_view_corns.config(state="normal")
            self.btn_calibrate.config(state="normal")
            self.btn_vis_reproj_error.config(state="normal")
            self.btn_vis_dist.config(state="normal")
            self.btn_save.config(state="normal")

            self.lbl_num_files.config(text=f'{len(self.files):d} files')
            self.lbl_corns_found.config(text='All corners found')
            self.lbl_cam_calibrated.config(text=f'Average reprojection error: {self.avg_reproj_error:.2f} pixels')

    def clear_reproj_errors(self):
        for name, val in zip(self.var_reproj_name, self.var_reproj_val):
            name.set('')
            val.set('')

    def close(self):
        """
        Closes all windows
        """
        # Close Sofast window
        self.root.destroy()


if __name__ == '__main__':
    CalibrationGUI()
