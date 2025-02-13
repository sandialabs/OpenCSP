"""
GUI used to load previously captured calibration images and calibrate a
machine vision camera.
"""

import os
import tkinter
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter import messagebox

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import convolve2d

import opencsp.app.camera_calibration.lib.calibration_camera as cc
import opencsp.app.camera_calibration.lib.image_processing as ip
from opencsp.app.camera_calibration.lib.ViewAnnotatedImages import ViewAnnotatedImages
import opencsp.app.sofast.lib.spatial_processing as sp
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.tk_tools as tkt


class CalibrationGUI:
    """
    A graphical user interface (GUI) for calibrating a machine vision camera.

    This class provides a user-friendly interface to load previously captured
    calibration images, find checkerboard corners, and calibrate the camera
    using the selected images. It also allows users to visualize the
    reprojection error and save the calibrated camera parameters.

    The GUI includes options to select images, find corners, view found
    corners, calibrate the camera, visualize distortion, and save the camera
    configuration.
    """

    # "ChatGPT 4o-mini" assisted with generating this docstring.
    def __init__(self):
        """Initializes GUI window"""
        # Create tkinter object
        self._root = tkt.window()

        # Set title
        self._root.title("Camera Calibration")

        # Set size of GUI
        self._root.geometry("550x480+200+100")

        # Add all buttons/widgets to window
        self._create_layout()

        # Set flags
        self._max_image_dimension = 2000  # Max image dimension that we give to OpenCV find_corners function
        self._files_slected = False
        self._images_loaded = False
        self._camera_calibrated = False
        self._downsample_factor: int = None  # Downsample factor for images that are too large for OpenCV

        # Initialize variables
        self._files: list[str]
        self._images: list[np.ndarray]
        self._used_file_names: list[str]
        self._p_object: list[Vxyz]
        self._p_image: list[Vxy]
        self._img_size_xy: tuple[int, int]
        self._img_size_xy_native: tuple[int, int]
        self._camera: Camera
        self._r_cam_object: list[Rotation]
        self._v_cam_object_cam: list[Vxyz]
        self._avg_reproj_error: list[float]
        self._reproj_errors: list[float]

        # Format buttons
        self._enable_btns()

        # Run window infinitely
        self._root.mainloop()

    def _create_layout(self):
        # BUTTONS / ENTRIES
        r = 0

        # Name of camera input
        self.var_cam_name = tkinter.StringVar(value="Camera")
        lbl_cam_name = tkinter.Label(self._root, text="Camera Name:", font=("calibre", 10, "bold"))
        entry_cam_name = tkinter.Entry(
            self._root, textvariable=self.var_cam_name, font=("calibre", 10, "normal"), width=40
        )

        lbl_cam_name.grid(row=r, column=0, pady=2, padx=2, sticky="nsw")
        entry_cam_name.grid(row=r, column=1, pady=2, padx=2, sticky="nsw")
        r += 1

        # Number of points input
        self.var_pts_x = tkinter.IntVar(value=18)
        self.var_pts_y = tkinter.IntVar(value=23)
        lbl_pts_x = tkinter.Label(self._root, text="Number of grid x points:", font=("calibre", 10, "bold"))
        lbl_pts_y = tkinter.Label(self._root, text="Number of grid y points:", font=("calibre", 10, "bold"))
        entry_pts_x = tkinter.Entry(self._root, textvariable=self.var_pts_x, font=("calibre", 10, "normal"), width=10)
        entry_pts_y = tkinter.Entry(self._root, textvariable=self.var_pts_y, font=("calibre", 10, "normal"), width=10)

        lbl_pts_x.grid(row=r, column=0, pady=2, padx=2, sticky="nsw")
        entry_pts_x.grid(row=r, column=1, pady=2, padx=2, sticky="nsw")
        r += 1
        lbl_pts_y.grid(row=r, column=0, pady=2, padx=2, sticky="nsw")
        entry_pts_y.grid(row=r, column=1, pady=2, padx=2, sticky="nsw")
        r += 1

        # Select images button
        self.btn_select_ims = tkinter.Button(self._root, text="Select Images", command=self._select_images)
        self.btn_select_ims.grid(row=r, column=0, pady=2, padx=2, sticky="nesw")
        r += 1

        # Find corners button
        self.btn_find_corns = tkinter.Button(self._root, text="Find Corners", command=self._find_corners)
        self.btn_find_corns.grid(row=r, column=0, pady=2, padx=2, sticky="nesw")
        r += 1

        # View annotated images button
        self.btn_view_corns = tkinter.Button(self._root, text="View Found Corners", command=self._view_found_corners)
        self.btn_view_corns.grid(row=r, column=0, pady=2, padx=2, sticky="nesw")
        r += 1

        # Calibrate button
        self.btn_calibrate = tkinter.Button(self._root, text="Calibrate Camera", command=self._calibrate_camera)
        self.btn_calibrate.grid(row=r, column=0, pady=2, padx=2, sticky="nesw")
        r += 1

        # Show reprojection error button
        self.btn_vis_reproj_error = tkinter.Button(
            self._root, text="Show reprojection error", command=self._show_reproj_error
        )
        self.btn_vis_reproj_error.grid(row=r, column=0, pady=2, padx=2, sticky="nesw")
        r += 1

        # Visualize distortion
        self.btn_vis_dist = tkinter.Button(self._root, text="Visualize distortion", command=self._visualize_dist)
        self.btn_vis_dist.grid(row=r, column=0, pady=2, padx=2, sticky="nesw")
        r += 1

        # Save camera button
        self.btn_save = tkinter.Button(self._root, text="Save Camera", command=self._save_camera)
        self.btn_save.grid(row=r, column=0, pady=2, padx=2, sticky="nesw")
        r += 1

        # Close button
        self.btn_close = tkinter.Button(self._root, text="Close", command=self._close)
        self.btn_close.grid(row=r, column=0, pady=2, padx=2, sticky="nesw")
        r += 1

        # Reprojection error labels
        lbl = tkinter.Label(text="Reprojection Error", borderwidth=1, relief="solid", font=("calibre", 10, "bold"))
        lbl.grid(row=r, column=0, sticky="nsew", pady=(20, 0))
        lbl = tkinter.Label(text="Image Name", width=40, borderwidth=1, relief="solid", font=("calibre", 10, "bold"))
        lbl.grid(row=r, column=1, sticky="nsw", pady=(20, 0))
        r += 1

        self.var_reproj_name = []
        self.var_reproj_val = []
        for _ in range(5):
            var_val = tkinter.StringVar(value="")
            var_name = tkinter.StringVar(value="")

            self.var_reproj_val.append(var_val)
            self.var_reproj_name.append(var_name)

            lbl_val = tkinter.Label(textvariable=var_val, borderwidth=1, relief="solid", font=("calibre", 10, "normal"))
            lbl_name = tkinter.Label(
                textvariable=var_name, borderwidth=1, width=40, relief="solid", font=("calibre", 10, "normal")
            )

            lbl_val.grid(row=r, column=0, sticky="nsew")
            lbl_name.grid(row=r, column=1, sticky="nsw")

            r += 1

        # LABELS
        r = 3

        # Selected files label
        self.lbl_num_files = tkinter.Label(self._root, font=("calibre", 10, "bold"))
        self.lbl_num_files.grid(row=r, column=1, pady=2, padx=2, sticky="nsw")
        r += 1

        # Corners found label
        self.lbl_corns_found = tkinter.Label(self._root, font=("calibre", 10, "bold"))
        self.lbl_corns_found.grid(row=r, column=1, pady=2, padx=2, sticky="nsw")
        r += 1

        # Camera calibrated label
        r += 1
        self.lbl_cam_calibrated = tkinter.Label(self._root, font=("calibre", 10, "bold"))
        self.lbl_cam_calibrated.grid(row=r, column=1, pady=2, padx=2, sticky="nsw")
        r += 1

    def _select_images(self):
        try:
            # Asks user to select files
            filetypes = [
                ("all files", ".*"),
                ("PNG files", "*.png"),
                ("JPG files", ("*.jpg", "*.jpeg")),
                ("TIF files", ("*.tif", "*.tiff")),
                ("GIF files", "*.gif"),
            ]
            files = askopenfilename(filetypes=filetypes, title="Select image files", multiple=True)

            if len(files) != "":
                # Save files
                self._files = files

                # Update flags
                self._files_slected = True
                self._images_loaded = False
                self._camera_calibrated = False

                # Clear data
                self._images = []
                self._used_file_names = []
                self._p_object = []
                self._p_image = []
                self._img_size_xy = []
                self._camera = None
                self._r_cam_object = []
                self._v_cam_object_cam = []
                self._avg_reproj_error = []
                self._reproj_errors = []

                # Format buttons
                self._enable_btns()
        except Exception as error:
            messagebox.showerror("Error with Selecting images", str(error))

    def _get_n_checkerboard_pts(self):
        return (self.var_pts_x.get(), self.var_pts_y.get())

    def _find_corners(self):
        try:
            # Get number checkerboard points
            npts = self._get_n_checkerboard_pts()

            # Reset data objects
            self._p_object = []
            self._p_image = []
            self._images = []
            self._used_file_names = []
            self._downsample_factor = None
            self._img_size_xy = None
            self._img_size_xy_native = None

            # Find corners
            for file in self._files:
                # Get file name
                file_name = os.path.basename(file)

                # Update progress
                print("Processing:", file_name, flush=True)

                # Load images
                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

                # Convert to uint8 for OpenCV camera calibration function
                img = img.astype("uint8")

                if self._img_size_xy_native is None:
                    # Define native xy shape
                    self._img_size_xy_native = (img.shape[1], img.shape[0])
                else:
                    # Check that image shape is consistent with first image
                    if self._img_size_xy_native != (img.shape[1], img.shape[0]):
                        raise ValueError(
                            f"Image {file_name:s} has (x,y) shape ({img.shape[1]},{img.shape[0]}) but was expecting shape {self._img_size_xy_native}"
                        )

                if self._downsample_factor is None:
                    # Calculate downsample factor if image is too large for OpenCV algorithm
                    if max(img.shape) > self._max_image_dimension:
                        self._downsample_factor = int(np.ceil(max(img.shape) / self._max_image_dimension))
                    else:
                        self._downsample_factor = int(1)

                if self._downsample_factor != 1:
                    # Downsample if necessary
                    img = self._downsample_image(img)

                if self._img_size_xy is None:
                    # Define first image downsampled shape
                    self._img_size_xy = (img.shape[1], img.shape[0])

                # Find checkerboard corners
                p_object, p_image = ip.find_checkerboard_corners(npts, img)
                if (p_object is None) or (p_image is None):
                    print(f"Could not find corners in image: {file_name:s}.")
                    continue

                # Save image, filename, and found corners
                self._images.append(img)
                self._used_file_names.append(file_name)

                self._p_object.append(p_object)
                self._p_image.append(p_image)

            # Update flags
            self._images_loaded = True
            self._camera_calibrated = False

            # Clear any calibrated camera
            self._camera = None
            self._r_cam_object = []
            self._v_cam_object_cam = []
            self._avg_reproj_error = []
            self._reproj_errors = []

            # Format buttons
            self._enable_btns()

        # Handle errors
        except Exception as error:
            messagebox.showerror("Find Corners Error", str(error))

    def _view_found_corners(self):
        try:
            # Create new window
            root_corns = tkt.window(self._root, TopLevel=True)

            # Get number checkerboard points
            npts = self._get_n_checkerboard_pts()

            # Annotate images
            ims = []
            for idx, image in enumerate(self._images):
                # Create RGB image
                im3 = np.concatenate([image[..., np.newaxis]] * 3, axis=2)
                # Annotate image
                ip.annotate_found_corners(npts, im3, self._p_image[idx])
                # Save image
                ims.append(im3)

            # View corners
            ViewAnnotatedImages(root_corns, ims, self._used_file_names)
        except Exception as error:
            messagebox.showerror("Error with viewing found corners", str(error))

    def _calibrate_camera(self):
        try:
            # Get camera name
            cam_name = self.var_cam_name.get()

            # Calibrate camera
            (self._camera, self._r_cam_object, self._v_cam_object_cam, self._avg_reproj_error) = cc.calibrate_camera(
                self._p_object, self._p_image, self._img_size_xy, cam_name
            )

            # Calculate reprojection error for each image
            self._reproj_errors = []
            for r_cam, v_cam, p_object, p_image in zip(
                self._r_cam_object, self._v_cam_object_cam, self._p_object, self._p_image
            ):
                error = sp.reprojection_error(self._camera, p_object, p_image, r_cam, v_cam)  # RMS pixels
                self._reproj_errors.append(error)  # RMS pixels

            # Find five images with highest reprojection errors
            idxs = np.flip(np.argsort(self._reproj_errors))[: len(self.var_reproj_name)]
            for idx, name, val in zip(idxs, self.var_reproj_name, self.var_reproj_val):
                name.set(self._used_file_names[idx])
                val.set(f"{self._reproj_errors[idx]:.2f}")

            # NOTE: self._downsample_factor = 1 if no downsampling occured
            # Update image shape
            self._camera.image_shape_xy = self._img_size_xy_native
            # Update intrinsic matrix non-zero and non-unity values
            self._camera.intrinsic_mat[0, 0] *= self._downsample_factor
            self._camera.intrinsic_mat[1, 1] *= self._downsample_factor
            self._camera.intrinsic_mat[0, 2] *= self._downsample_factor
            self._camera.intrinsic_mat[1, 2] *= self._downsample_factor

            # Update flags
            self._camera_calibrated = True

            # Format buttons
            self._enable_btns()

        # Handle errors
        except Exception as error:
            messagebox.showerror("Error calibrating camera", str(error))

    def _show_reproj_error(self):
        try:
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(self._reproj_errors, "o-")
            ax.set_xlim(-0.5, len(self._reproj_errors) + 0.5)
            ax.xaxis.set_ticks(np.arange(len(self._reproj_errors)))
            ax.grid()
            ax.set_ylabel("Reprojection Error (Pixels RMS)")
            ax.set_xlabel("Image Index Number")
            plt.show(block=False)
        except Exception as error:
            messagebox.showerror("Error with showing reprojection errors", str(error))

    def _visualize_dist(self):
        try:
            ax1 = plt.subplot(131)
            ax2 = plt.subplot(132)
            ax3 = plt.subplot(133)
            cc.view_distortion(self._camera, ax1, ax2, ax3)
            plt.show(block=False)
        except Exception as error:
            messagebox.showerror("Error with visualizing distortion", str(error))

    def _save_camera(self):
        try:
            file = asksaveasfilename(defaultextension=".h5", filetypes=[("HDF5 File", "*.h5")])
            if file != "":
                self._camera.save_to_hdf(file)
        except Exception as error:
            messagebox.showerror("Error with saving camera", str(error))

    def _enable_btns(self):
        if not self._files_slected:
            # No files selected
            self.btn_find_corns.config(state="disabled")
            self.btn_view_corns.config(state="disabled")
            self.btn_calibrate.config(state="disabled")
            self.btn_vis_reproj_error.config(state="disabled")
            self.btn_vis_dist.config(state="disabled")
            self.btn_save.config(state="disabled")

            self.lbl_num_files.config(text="0 files")
            self.lbl_corns_found.config(text="Corners not found")
            self.lbl_cam_calibrated.config(text="Not calibrated")

            self._clear_reproj_errors()
        elif not self._images_loaded:
            # Files selected, but not not loaded/processed
            self.btn_find_corns.config(state="normal")
            self.btn_view_corns.config(state="disabled")
            self.btn_calibrate.config(state="disabled")
            self.btn_vis_reproj_error.config(state="disabled")
            self.btn_vis_dist.config(state="disabled")
            self.btn_save.config(state="disabled")

            self.lbl_num_files.config(text=f"{len(self._files):d} files")
            self.lbl_corns_found.config(text="Corners not found")
            self.lbl_cam_calibrated.config(text="Not calibrated")

            self._clear_reproj_errors()
        elif self._camera is None:
            # Files selected and processed, but camera not calibrated
            self.btn_find_corns.config(state="normal")
            self.btn_view_corns.config(state="normal")
            self.btn_calibrate.config(state="normal")
            self.btn_vis_reproj_error.config(state="disabled")
            self.btn_vis_dist.config(state="disabled")
            self.btn_save.config(state="disabled")

            self.lbl_num_files.config(text=f"{len(self._files):d} files")
            self.lbl_corns_found.config(text="All corners found")
            self.lbl_cam_calibrated.config(text="Not calibrated")

            self._clear_reproj_errors()
        else:
            # Camera calibrated
            self.btn_find_corns.config(state="normal")
            self.btn_view_corns.config(state="normal")
            self.btn_calibrate.config(state="normal")
            self.btn_vis_reproj_error.config(state="normal")
            self.btn_vis_dist.config(state="normal")
            self.btn_save.config(state="normal")

            self.lbl_num_files.config(text=f"{len(self._files):d} files")
            self.lbl_corns_found.config(text="All corners found")
            self.lbl_cam_calibrated.config(text=f"Average reprojection error: {self._avg_reproj_error:.2f} pixels")

    def _clear_reproj_errors(self):
        for name, val in zip(self.var_reproj_name, self.var_reproj_val):
            name.set("")
            val.set("")

    def _downsample_image(self, image: np.ndarray) -> np.ndarray:
        # Create square anti-aliasing filter
        n = self._downsample_factor
        dtype = image.dtype
        ker = np.ones((n, n)) / (n**2)

        # Convert to grayscale
        if np.ndim(image) == 3:
            image = image.mean(2)
        elif np.ndim(image) != 2:
            raise ValueError(f"Input image shape must be 2 or 3 dimensions but was shape, {image.shape}")

        # Apply anti-aliasing filter
        image = convolve2d(image, ker)

        # Downsample
        image = image[::n, ::n]

        # Convert to input dtype
        return image.astype(dtype)

    def _close(self):
        # Close Sofast window
        self._root.destroy()


if __name__ == "__main__":
    CalibrationGUI()
