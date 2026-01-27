example_process_single_facet_README.txt:
========================================
Example SOFAST data processing for a single facet measurement.  
Given a stored measurement file from a SOFAST data acquisition, process the file to 
construct a slope map and generate the standard plot output suite.

This file can be run in several different modes, explained below.

Run Pytest
----------

To run this as a pytest on the built-in input:
   1. cd to the OpenCSP code directory.
   2. cd to the example subdirectory.
   3. Execute pytest:
          pytest
       or
          pytest sofast_fringe\single_facet\example_process_single_facet.py


Default Run on Built-In Data
----------------------------

To run this on the default built-in input:
   1. cd to the directory containing the script "example_process_single_facet.py".
   2. Run the script:
              python example_process_single_facet.py --verbose
       The "--verbose" flag is optional.

This runs the code on data that is built into the repository.  To avoid bloating
the repository, this input data has been downsampled to reduce its size.


Run on Other Data
-----------------

The built-in data described above enables automatic testing to verify code execution, 
but the data is heavily downsampled and thus not representative of realistic output.
Also the built-in input and output files are in obscure locations that don't correspond 
to how one would organize data when using SOFAST in practice.

The example_process_single_facet.py file provides a -s option that points to a settings 
control file.  This text file is easy to edit, and enables you to point to your preferred 
location of input data, and also direct output to your preferred location.  You can also 
control program execution and output simply by editing the settings file.  This avoids 
the need to modify source code simply to run a new problem (which is highly discouraged).

You can easily create a settings file for your data, with your run preferences, 
and then execute it without modifying the example_process_single_facet.py file or 
other code.

To see example settings files, see:

    <OpenCSP_code_dir>\example\sofast_fringe\single_facet\

There are several files there with a ".ini" extension, which execute the code in 
different scenarios.  These scenarios include information exploration on your local 
computer, or more organized measurement campaigns either on your local computer 
or in a network server environment.

OpenCSP provides full-size example data which can be used for both of these scenarios, 
enabling you to learn how the code should run and what its output should look like.  
This experience will give you familiarity with the process when you run the code on 
your own measurement data.  Below we explain how to use these settings and example 
data files for both informal exploration and a more structured measurement campaign.


Local Exploration
-----------------

For an example settings file designed for exploration, see:

<?? PUT Ctemp SCENARIO EXPLANATION HERE ??>

    dir:   <OpenCSP_code_dir>\example\sofast_fringe\single_facet\
    file:  ????_Ctemp.ini

Note this files has a "ctemp" suffix, indicating that they assume a local directory "C:\ctemp\"
which contains the test data published on OpenCSP.  This enables you to execute exploratory runs
on your local computer, without any modifications to the script.


Analyzing Data with an Archival Directory Scheme
------------------------------------------------

To run the single-facet SOFAST processing on a directory structure designed for ongoing measurement 
and archival of results, see the file:

    dir:   <OpenCSP_code_dir>\example\sofast_fringe\single_facet\
    file:  20250818_163443_SNLTF-A_OLSLrsqw_p001_default_process_single_facet_settings_Q.ini

This file contains instructions explaining how to configure the directories and data files, and then 
how to execute the example_process_single_facet.py file on that scenario.

For variations of the default scenario that run faster, against different reference mirrors, etc, 
see the variation files:

    dir:   <OpenCSP_code_dir>\example\sofast_fringe\single_facet\
    file:  20250818_163443_SNLTF-A_OLSLrsqw_p002_fast_process_single_facet_settings_Q.ini
    file:  20250818_163443_SNLTF-A_OLSLrsqw_p003_ref_25m_process_single_facet_settings_Q.ini
    file:  20250818_163443_SNLTF-A_OLSLrsqw_p004_ref_plano_process_single_facet_settings_Q.ini

Note that these files each have a different "_p00x_xx_" substring, which is the "post_process_id"
denoting a particular set of processing and output settings.  This provides a means for running 
the code under different settings, and then comparing the results.

Also note that the files have a "Q" suffix, indicating that they assume a mapped drive "Q:".  
This enables you to put the input and output data in your preferred location, map the Q: drive 
letter to that location, and then execute the example script without modification.


Analyzing Your New Data
-----------------------


<?? DELETE THE BELOW? ??>

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

For an example settings file designed for measuring and logging results over time, see:
    <OpenCSP_code_dir>\example\sofast_fringe\single_facet\20250818_163443_SNLTF-A_OLSLrsqw_p001_default_process_single_facet_settings_Q.ini

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
