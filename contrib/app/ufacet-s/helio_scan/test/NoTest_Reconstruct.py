"""
Testing the OpenCV 3-d reconstruction on tracked heliostat corners.



"""

import glob
import numpy as np
import os
from multiprocessing import Pool
import subprocess

import opencsp.common.lib.tool.file_tools as ft
import lib.ufacet_heliostat_3d_analysis as uh3a
import lib.DEPRECATED_specifications as Dspec  # ?? SCAFFOLDING RCB -- TEMPORARY
import lib.DEPRECATED_utils as utils  # ?? SCAFFOLDING RCB -- TEMPORARY


"""
Original reconstruction code, with test annotations added.


Here are some notes from my first test, which was executing the program from within the "manual_shell_example" directory:

        After typing in the following environment setup line:

            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PROJECTS_DIR/netpub/opencv/ceres-solver/1.14.0/lib64:$PROJECTS_DIR/netpub/opencv/gflags/lib:$PROJECTS_DIR/netpub/opencv/glog/lib64:$PROJECTS_DIR/netpub/opencv/gflags/lib

        Typing "./reconstruct_main.out" resulted in the executable printing a help message.
        [Note:  This help message was in error, because it did not list all the program command-line arguments.]

        After consulting the C++ source code "main.cpp,"  I concluded that the proper arguments are:

            reconstruct main.out input_patt_to_file fx fy cx cy heliostat_name output_dir/

        Thus the following line worked successfully:

        ./reconstruct_main.out input/9E1_tracks.txt 2868.1 2875.9 3840 2160 9E1 output/



"""


class Reconstruct:
    def __init__(
        self,
        executable_path,  # where to find reconstruct executable
        confirm_distort_str,  # whether corner files reflect confirmed or projected, and distorted or undistorted.  Example: 'confirmed_undistorted'
        input_heliostat_2d_corners_dir,  # where to find per-heliostat corner track files
        output_heliostat_3d_dir,  # where to save reconstructed 3-d heliostat models
        output_evaluation_plot_dir,  # where to save output plots.
        output_evaluation_csv_dir,  # where to save output csv files.
        cam_matrix,  # camera matrix
        heliostat_3d_file_suffix,  # Suffix of 3-d heliostat file.
        theoretical_heliostat_dir_body_ext,  # Heliostat corners file.
    ):
        self.executable_path = executable_path
        self.confirm_distort_str = confirm_distort_str
        self.input_heliostat_2d_corners_dir = input_heliostat_2d_corners_dir
        self.output_heliostat_3d_dir = output_heliostat_3d_dir
        self.output_evaluation_plot_dir = output_evaluation_plot_dir
        self.output_evaluation_csv_dir = output_evaluation_csv_dir
        self.cam_matrix = cam_matrix
        self.heliostat_3d_file_suffix = heliostat_3d_file_suffix
        self.theoretical_heliostat_dir_body_ext = theoretical_heliostat_dir_body_ext

        all_files = glob.glob(input_heliostat_2d_corners_dir + "*")
        self.files = []
        for file in all_files:
            if self.heliostat_3d_file_suffix in file:
                self.files.append(file)

        # Supporting information.  # ?? SCAFFOLDING RCB -- SHOULD BE READ FROM ELSEWHERE.
        # Solar field parameters.
        self.specifications = Dspec.nsttf_specifications()  # ?? SCAFFOLDING RCB -- MAKE THIS GENERAL

        self.heliostat_theoretical = uh3a.read_txt_file_to_heliostat(
            self.theoretical_heliostat_dir_body_ext, self.specifications
        )

    def execute_reconstruction(self, file):
        hel_name = self.heliostat_name_given_heliostat_2d_corner_trajectories_dir_body_ext(file)
        # hel_name   = file.split('/')[-1].split('_')[0]

        # Perform the reconstruction.
        print("In execute_reconstruction(), calling call_executable() for heliostat " + hel_name + "...")
        self.call_executable(file)
        print("In execute_reconstruction(), call_executable() for heliostat " + hel_name + " finished.")

        # We plan to recompile the C++ executable to have more fine-grain control over its output filename, but not today.
        # So rename the output file to match our naming standard.
        print("In execute_reconstruction(), renaming output file for heliostat " + hel_name + "...")
        executable_output_body_ext = hel_name + "_reconstructed.txt"
        executable_output_dir_body_ext = os.path.join(self.output_heliostat_3d_dir, executable_output_body_ext)
        heliostat_3d_dir_body_ext = os.path.join(
            self.output_heliostat_3d_dir, hel_name + "_" + self.confirm_distort_str + "_corners_3d.txt"
        )
        ft.rename_file(executable_output_dir_body_ext, heliostat_3d_dir_body_ext)

        print("In execute_reconstruction(), calling generate_plots() for heliostat " + hel_name + "...")
        uh3a.generate_plots(
            heliostat_3d_dir_body_ext, output_evaluation_plot_dir, self.specifications, self.heliostat_theoretical
        )
        print("In execute_reconstruction(), generate_plots() for heliostat " + hel_name + " finished.")

        return heliostat_3d_dir_body_ext

    def call_executable(self, file):
        hel_name = self.heliostat_name_given_heliostat_2d_corner_trajectories_dir_body_ext(file)
        # hel_name = file.split('/')[-1].split('_')[0]
        fx = self.cam_matrix[0][0]
        fy = self.cam_matrix[1][1]
        cx = self.cam_matrix[0][2]
        cy = self.cam_matrix[1][2]

        print("In call_executable(), self.executable_path =", self.executable_path)
        print("In call_executable(), file =", file)
        print("In call_executable(), str(fx)   =", str(fx))
        print("In call_executable(), str(fy)   =", str(fy))
        print("In call_executable(), str(cx)   =", str(cx))
        print("In call_executable(), str(cy)   =", str(cy))
        print("In call_executable(), heliostat =", hel_name)
        print("In call_executable(), self.output_heliostat_3d_dir =", self.output_heliostat_3d_dir)

        print("In call_executable(), calling executable for heliostat " + hel_name + "...")

        proc = subprocess.Popen(
            [self.executable_path, file, str(fx), str(fy), str(cx), str(cy), hel_name, self.output_heliostat_3d_dir]
        )
        proc.wait()

        print("In call_executable(), executable for heliostat " + hel_name + " finished.")

    def perform_reconstruction(self, single_execution=True):
        print("self.files =", self.files)
        print("In perform_reconstruction(), starting reconstruction...")
        if single_execution:
            heliostat_3d_dir_body_ext_list = []
            for file in self.files:
                heliostat_3d_dir_body_ext_list.append(self.execute_reconstruction(file))
        else:
            with Pool(36) as pool:
                heliostat_3d_dir_body_ext_list = pool.map(self.execute_reconstruction, self.files)
        print("In perform_reconstruction() reconstruction finished.")

        print("heliostat_3d_dir_body_ext_list = ", heliostat_3d_dir_body_ext_list)

        print("In perform_reconstruction(), starting csv file generation...")
        output_evaluation_csv_dir_2 = (
            self.output_evaluation_csv_dir + "/"
        )  # ?? SCAFFOLDING RCB -- MAKE THIS PLATFORM INDEPENDENT.
        uh3a.generate_csv(
            heliostat_3d_dir_body_ext_list, output_evaluation_csv_dir_2, self.specifications, self.heliostat_theoretical
        )
        print("In perform_reconstruction() csv files finished.")

    def heliostat_name_given_heliostat_2d_corner_trajectories_dir_body_ext(
        self, heliostat_2d_corner_trajectories_dir_body_ext
    ):  # ?? SCAFFOLDING RCB -- MAKE THIS GENERAL, CORRECT, ERROR-CHECKING.  (IT'S LATE, AND I'M OUT OF TIME.)
        (
            heliostat_2d_corner_trajectories_dir,
            heliostat_2d_corner_trajectories_body,
            heliostat_2d_corner_trajectories_ext,
        ) = ft.path_components(heliostat_2d_corner_trajectories_dir_body_ext)
        tokens = heliostat_2d_corner_trajectories_body.split("_")
        trajectories_str = tokens[-1]
        two_d_str = tokens[-2]
        corner_str = tokens[-3]
        distorted_str = tokens[-4]
        confirmed_str = tokens[-5]
        name_str = tokens[-6]
        return name_str


if __name__ == "__main__":
    executable_path = (
        home_dir() + "Code/ufacet_code/Reconstruct/bin/reconstruct_main.out"
    )  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS.  PASS IN?

    # # 02_StandardExample
    # confirm_distort_str                = 'confirmed_undistorted'
    # input_heliostat_2d_corners_dir     = home_dir() + 'Code/ufacet_sampledata/Test_Reconstruct_Data/02_StandardExample/input_merged_tracks/'
    # output_heliostat_3d_dir            = home_dir() + 'Code/ufacet_sampledata/Test_Reconstruct_Data/02_StandardExample/output_reconstructed_heliostats/'
    # output_evaluation_plot_dir         = home_dir() + 'Code/ufacet_sampledata/Test_Reconstruct_Data/02_StandardExample/output_evaluation_plots/'
    # output_evaluation_csv_dir          = home_dir() + 'Code/ufacet_sampledata/Test_Reconstruct_Data/02_StandardExample/output_evaluation_csv_files/'
    # cam_matrix                         = utils.CameraMatrix  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
    # heliostat_3d_file_suffix           = '_' + confirm_distort_str + '_corner_2d_trajectories.txt'
    # theoretical_heliostat_dir_body_ext = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/010_HeliostatModel/data/NSTTF_Facet_Corners.csv'  # ?? SCAFFOLDING RCB -- MOVE THIS INTO TEST DATA.

    # # 01_Study_5W1
    # confirm_distort_str                = 'confirmed_undistorted'
    # input_heliostat_2d_corners_dir     = experiment_dir() + '2020-12-03_FastScan1/9_Analysis/2021-09-21_ReconstructionStudy/01_Study_5W1_solo/input_merged_tracks/'
    # output_heliostat_3d_dir            = experiment_dir() + '2020-12-03_FastScan1/9_Analysis/2021-09-21_ReconstructionStudy/01_Study_5W1_solo/output_reconstructed_heliostats/'
    # output_evaluation_plot_dir         = experiment_dir() + '2020-12-03_FastScan1/9_Analysis/2021-09-21_ReconstructionStudy/01_Study_5W1_solo/output_evaluation_plots/'
    # output_evaluation_csv_dir          = experiment_dir() + '2020-12-03_FastScan1/9_Analysis/2021-09-21_ReconstructionStudy/01_Study_5W1_solo/output_evaluation_csv_files/'
    # cam_matrix                         = utils.CameraMatrix  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
    # heliostat_3d_file_suffix           = '_' + confirm_distort_str + '_corner_2d_trajectories.txt'
    # theoretical_heliostat_dir_body_ext = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/010_HeliostatModel/data/NSTTF_Facet_Corners.csv'  # ?? SCAFFOLDING RCB -- MOVE THIS INTO TEST DATA.

    # 02_Study_5W1_9E9_11W1_14E3
    confirm_distort_str = "confirmed_undistorted"
    input_heliostat_2d_corners_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/9_Analysis/2021-09-21_ReconstructionStudy/02_Study_5W1_9E9_11W1_14E3_solo/input_merged_tracks/"
    )
    output_heliostat_3d_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/9_Analysis/2021-09-21_ReconstructionStudy/02_Study_5W1_9E9_11W1_14E3_solo/output_reconstructed_heliostats/"
    )
    output_evaluation_plot_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/9_Analysis/2021-09-21_ReconstructionStudy/02_Study_5W1_9E9_11W1_14E3_solo/output_evaluation_plots/"
    )
    output_evaluation_csv_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/9_Analysis/2021-09-21_ReconstructionStudy/02_Study_5W1_9E9_11W1_14E3_solo/output_evaluation_csv_files/"
    )
    cam_matrix = utils.CameraMatrix  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
    heliostat_3d_file_suffix = "_" + confirm_distort_str + "_corner_2d_trajectories.txt"
    theoretical_heliostat_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/010_HeliostatModel/data/NSTTF_Facet_Corners.csv"
    )  # ?? SCAFFOLDING RCB -- MOVE THIS INTO TEST DATA.

    rec = Reconstruct(
        executable_path,
        confirm_distort_str,
        input_heliostat_2d_corners_dir,
        output_heliostat_3d_dir,
        output_evaluation_plot_dir,
        output_evaluation_csv_dir,
        cam_matrix,
        heliostat_3d_file_suffix,
        theoretical_heliostat_dir_body_ext,
    )

    print("\n\nAt toplevel, calling perform_reconstruction()...")
    rec.perform_reconstruction(single_execution=False)  # True)
