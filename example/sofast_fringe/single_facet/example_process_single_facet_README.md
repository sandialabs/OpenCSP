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


Run on Published OpenCSP Example Data
-------------------------------------

The built-in data described above enables automatic testing to verify code execution, 
but the data is heavily downsampled and thus not representative of realistic output.
Also the built-in input and output files are in obscure locations that don't correspond 
to how one would organize data when using SOFAST in practice.

OpenCSP provides full-size example data which can be used for both of these scenarios, 
enabling you to learn how the code should run and what its output should look like.  
This experience will give you familiarity with the process when you run the code on 
your own measurement data.  The full-size example data shows how to run this example, 
generating realistic output.  It also clarifies input vs. output files.

To run this on the published OpenCSP full-size example input: 
   1. Copy the released OpenCSP data from:

          https://opencsp.sandia.gov
          --> How to Participate
          --> Access OpenCSP
          --> shared box.com link
          --> OpenCSP_Data
          --> OpenCSP_SampleData
          --> SofastFringe
          --> SingleFacet
          --> SofastFringeSingleFacetData_v1.0.zip

   2. Unzip and place the resulting enclosed "single_facet" directory within the 
      "sofast_fringe" directory, producing:

          C:\ctemp\OpenCSP_example_data\sofast_fringe\single_facet\

   3. After this, the "example_process_single_facet.py" script should run without modification, 
      by executing the following:
          5.1. Start a powershell.
          5.2. Ensure the OpenCSP virtual environment is activated.
          5.3. Navigate to the directory containing example_process_single_facet.py.
          5.4. Execute the following command:
                  python example_process_single_facet.py --verbose -s C:\ctemp\OpenCSP_example_data\sofast_fringe\single_facet\input\Results\OLSL\20250818_163443\20250818_163443_SNLTF-A_OLSLrsqw_p001_default_process_single_facet_settings_ctemp.ini
               The script should write to an "output" subdirectory placed next to the input directory.


Run on Other Data
-----------------

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
different scenarios.  These scenarios include default analysis, fast abbreviated 
analysis, comparison again alternative baseline mirrors, etc.

OpenCSP provides full-size example data which can be used for both of these scenarios, 
enabling you to learn how the code should run and what its output should look like.  
This experience will give you familiarity with the process when you run the code on 
your own measurement data.


Analyzing Data with an Archival Directory Scheme
------------------------------------------------

To run the single-facet SOFAST processing on a directory structure designed for ongoing measurement 
and archival of results, see the file:

    dir:   <OpenCSP_code_dir>\example\sofast_fringe\single_facet\
    file:  20250818_163443_SNLTF-A_OLSLrsqw_p001_default_process_single_facet_settings_ctemp.ini

This file contains instructions explaining how to configure the directories and data files, and then 
how to execute the example_process_single_facet.py file on that scenario.

For variations of the default scenario that run faster, against different reference mirrors, etc, 
see the variation files:

    dir:   <OpenCSP_code_dir>\example\sofast_fringe\single_facet\
    file:  20250818_163443_SNLTF-A_OLSLrsqw_p002_fast_process_single_facet_settings_ctemp.ini
    file:  20250818_163443_SNLTF-A_OLSLrsqw_p003_ref_25m_process_single_facet_settings_ctemp.ini
    file:  20250818_163443_SNLTF-A_OLSLrsqw_p004_ref_plano_process_single_facet_settings_ctemp.ini

Note that these files each have a different "_p00x_xx_" substring, which is the "post_process_id"
denoting a particular set of processing and output settings.  This provides a means for running 
the code under different settings, and then comparing the results.

Also note that the files have a "ctemp" suffix, indicating that they are designed to work with data placed in "C:\ctemp\OpenCSP_example_data\".  See the OpenCSP data README file for further inrformation.


Analyzing Your New Data
-----------------------

The -s option and settings.ini file enable you to put input data in your preferred location, and 
write output data to your preferred location, without the need to modify the script source code.
(Please avoid modifying the script for your individual problem.)

For an example settings file designed for measuring and logging results over time, see:

    <OpenCSP_code_dir>\example\sofast_fringe\single_facet\20250818_163443_SNLTF-A_OLSLrsqw_p001_default_process_single_facet_settings_ctemp.ini

If you replace "C:\ctemp\OpenCS_example_data\" with the path to your data, or a suitable mapped 
drive letter, then you should be able to run the example_process_sigle_facet.py script to analyze 
your data.  Note that the configuration file includes various parametres for comntrolling the 
analysis process, selecting desired output figures, setting ray tracing scenario parameters, etc.


For a detailed description of the algorithm and its input and output, see: 
    B. J. Smith, R. C. Brost, and B. G. Bean.
    Scene Reconstruction User Guide, Document Version 1.0.
    Sandia National Laboratories Report SAND2024-10625, August 2024.
    https://doi.org/10.2172/2463024 

Also available through OpenCSP_Documents; see https://opencsp.sandia.gov

========================================
