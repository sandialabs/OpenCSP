example_process_single_facet_README.txt:
========================================
Example SOFAST data processing for a single facet measurement.  
Given a stored measurement file from a SOFAST data acquisition, process the file to 
construct a slope map and generate the standard plot output suite.

To run this as a pytest on the built-in input:
   1. cd to the OpenCSP code directory.
   2. cd to the example subdirectory.
   3. Execute pytest:
          pytest
       or
          pytest sofast_fringe\example_process_single_facet.py

To run this on the default built-in input, omit the -s option:
   1. cd to the directory containing the script "example_process_single_facet.py".
   2. Run the script:
              python example_process_single_facet.py --verbose
       The "--verbose" flag is optional.'

To run this on new input, use the -s option and point to a settings control file.
For an example settings file, see:
    <OpenCSP_code_dir>\example\process_single_facet\example_process_single_facet_settings_ctemp.ini 

We recommend copying this file and placing it alongside the data you wish to run.
For example, to run the full-size OpenCSP example:'
   1. Create a directory "C:\ctemp\OpenCSP_ctemp\example_data_large" for holding example data.
   2. Create subdirectory "process_single_facet" 
   3. Create subsubdirectory "input", and fill it with the required input files:
         "alignment_points.csv"
         "camera.h5"
         "known_point_locations.csv"
         "point_pair_distances.csv"
         Subdirectory "aruco_marker_images" containing images of scene with Aruco markers.
   4. Copy the file "example_process_single_facet_settings_ctemp.ini" into the subdirectory
      "process_single_facet" made in step 2.'
   5. Launch a PowerShell and ensure the OpenCSP virtual environment is activated.
   6. cd to the directory containing the script "example_process_single_facet.py".
   7. Run the script, providing the -s option and pointing to the .ini file:
         python example_process_single_facet.py --verbose -s "C:\ctemp\OpenCSP_ctemp\example_data_large\process_single_facet\example_process_single_facet_settings_ctemp.ini" 
   8. The "--verbose" option generates additional status and calculation output.
   9. The output will be written to an "example_process_single_facet_output" subdirectory 
      created alongside the input directory.
      (This is to distinguish it from output from other examples within the directory.)

To run this calculation on your own data:
   A. Create a directory holding your input data.
   B. Copy the "example_process_single_facet_settings_ctemp.ini" file to a new name, such as 
      "My_Data_process_single_facet_settings.ini" and edit it to point to your data location.
   C. For the sake of example, suppose you place your data in "C:\ctemp\OpenCSP_ctemp\MyData",
      and also suppose you place your new "My_Data_process_single_facet_settings.ini" file
      in this directory.
      Then you can run the same calculation on your data by:
         a. cd to the directory containing the script "example_process_single_facet.py".
         b. Run the script, providing the -s option and pointing to your .ini file:
              python example_process_single_facet.py --verbose -s "C:\ctemp\OpenCSP_ctemp\MyData\My_Data_process_single_facet_settings_ctemp.ini" 
   D. The output will be written to the output subdirectory you specify in your .ini file.'

For a detailed description of the algorithm and its input and output, see: 
    B. J. Smith, R. C. Brost, and B. G. Bean.
    Scene Reconstruction User Guide, Document Version 1.0.
    Sandia National Laboratories Report SAND2024-10625, August 2024.
    https://doi.org/10.2172/2463024 

Also available through OpenCSP_Documents; see https://opencsp.sandia.gov

========================================
