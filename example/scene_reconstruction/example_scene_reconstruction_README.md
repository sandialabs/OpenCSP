example_scene_reconstruction_README.txt:
========================================
Example scene reconstruction calculation.  Given photos with Aruco markers, find marker
and camera 3-d positions.  See "example_scene_reconstruction_README.txt" for details.


To run this as a pytest on the built-in input:
   1. cd to the OpenCSP code directory.
   2. cd to the example subdirectory.
   3. Execute pytest:
          pytest
       or
          pytest scene_reconstruction\example_scene_reconstruction.py


To run this on the default built-in input:
   1. cd to the directory containing the script "example_scene_reconstruction.py".
   2. Run the script:
              python example_scene_reconstruction.py --verbose
       The "--verbose" flag is optional.


To run this on the published OpenCSP full-size example input: 
   1. Copy the released OpenCSP data from:

          https://opencsp.sandia.gov
          --> How to Participate
          --> Access OpenCSP
          --> shared box.com link
          --> OpenCSP_Data
          --> OpenCSP_SampleData
          --> SceneReconstruction
          --> SceneReconstructionData_v1.1.zip

   2. Unzip and place the enclosed directory within the directory:

          C:\ctemp\OpenCSP_example_data\enclosed_energy\

   3. This will produce a directory structure:

          C:\ctemp\OpenCSP_example_data\enclosed_energy\EnclosedEnergyData_v1.1\

   4. Then rename "EnclosedEnergyData_v1.1" to "input", producing:

          C:\ctemp\OpenCSP_example_data\enclosed_energy\input\

   5. After this, the "example_scene_reconstruction.py" script should run without modification, 
      by executing the following:
          5.1. Start a powershell.
          5.2. Ensure the OpenCSP virtual environment is activated.
          5.3. Navigate to the directory containing example_scene_reconstruction.py.
          5.4. Execute the following command:
                  python example_scene_reconstruction.py --verbose -s C:\ctemp\OpenCSP_example_data\scene_reconstruction\input\example_scene_reconstruction_settings_ctemp.ini
               The script should write to an "output" subdirectory placed next to the input directory.


To run this on your own new input:
   1. Set up your input directory to contain the required information.  See the published OpenCSP scene 
      reconstruction example for the files that are needed.  These include:
           "alignment_points.csv"
           "camera.h5"
           "known_point_locations.csv"
           "point_pair_distances.csv"
           Subdirectory "aruco_marker_images" containing images of scene with Aruco markers.

   2. Create a settings configuration file.  For an example settings file, see:
         <OpenCSP_code_dir>\example\scene_reconstruction\example_scene_reconstruction_settings_ctemp.ini 
      We suggest copying this file, renaming it, placing it alongside your data, and editing 
      it to point to your data and output locations.

   3. For the sake of example, suppose you place your data in "C:\ctemp\OpenCSP_ctemp\MyData",
      and also suppose you place your new "My_Data_scene_reconstruction_settings.ini" file
      in this directory.
      Then you can run the same calculation on your data by:
          3.1. Start a powershell.
          3.2. Ensure the OpenCSP virtual environment is activated.
          3.3. Navigate to the directory containing the script "example_scene_reconstruction.py".
          3.4. Run the script, providing the -s option and pointing to your .ini file:
                  python example_scene_reconstruction.py --verbose -s "C:\ctemp\OpenCSP_ctemp\MyData\My_Data_scene_reconstruction_settings_ctemp.ini" 
               The script shouyld write to the output directory specified in your .ini file.'


For a detailed description of the algorithm and its input and output, see: 
    B. J. Smith, R. C. Brost, and B. G. Bean.
    Scene Reconstruction User Guide, Document Version 1.0.
    Sandia National Laboratories Report SAND2024-10625, August 2024.
    https://doi.org/10.2172/2463024 

Also available through OpenCSP_Documents; see https://opencsp.sandia.gov

========================================
