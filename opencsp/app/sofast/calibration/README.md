# Open CSP directory app/sofast/calibration/
Contains scripts and tools used to calibrate the SOFAST pysical setup.

## lib/
Library of photogrammetric algorithms used for SOFAST calibration.

## Scripts:
The following scripts are meant to be run in a terminal.
- **find_aruco_markers_in_image.py** - Annotates images with aruco markers present and saves as separate image file. Useful to run on Aruco dataset to ensure all markers are found as expected.
- **genrate_aruco_images.py** - Generates a series of Aruco markers as PNG images using given parameters.
- **run_full_calibration.py** - Runs a ful photogrammetric SOFAST calibration; outputs saved calibration files. Requires user inputs to be in the correct format as described below.

## File structure for full calibration:
The calibration scripts expect the following file naming/location conventions to be in place:

- **_base direcory/_** - This is the containing folder of all the input/output calibration files.
   - **data_input/** - All input files are stored within this directory.
      - **aruco_marker_images** - Directory contianing set of captured image sets of aruco markers set up around screen calibration points, around Sofast area, and in view of Sofast camera.
         - image_1.jpg
         - image_2.jpg
         - ...
      - **screen_shape_sofast_measurements/** - Contains the Sofast measurements of the screen used for screen shape measurement. Files must start with "pose_".
         - pose_1.h5
         - pose_2.h5
         - pose_3.h5
         - ...
      - **point_pair_distances.csv** - CSV file contianing aruco marker ID A, ID B, and distance between IDs A and B aranged in three columns with one header line.
      - **screen_calibration_point_pairs.csv** - CSV file containing the screen calibration fiducial point ID, the corresponding aruco marker ID (if present, otherwise blank), and the approximate X/Y/Z positions of the points aranged in three columns with one header line.
      - **camera_aruco_marker.5** - The  file describing the calibration of the camera used to capture the aruco markers.
      - **camera_screen_shape.h5** - The file describing the calibration of the camera used to capture the screen shape measurements.
      - **camera_sofast.h5** - The camera describing the calibration of the camera used as part of the Sofast installation.
      - **image_projection.h5** The image projection file used in the Sofast installation.
      - **image_sofast_camera.png** - The image captured by the Sofast camera while the aruco markers were in place within its field of view.
   - **data_output/** - All output files are saved to this directory.
      - **calibrated_corner_locations_absolute.csv** - The xyz positions of all corners of all aruco markers within the scene with absolute scale and alignment; units of meters.
      - **calibrated_corner_locations_relative.csv** - The relative positions of all corners of all aruco markers within the scene. Positions are not to scale nor aligned to any coordinate axis.
      - **camera_rvec_tvec.csv** - The rotation and translation vectors of the camera with respect to the screen.
      - **screen_distortion_data.h5** - The distortion data generated during the screen shape calibration.
      - **_Output display HDF5 file_** - The output Sofast display calibration file.
