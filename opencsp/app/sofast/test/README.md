# Sofast unit test suite

## 1. Generate downsampled measurement files
Run *generate_downsampled_dataset.py* to create downsampled versions of Sofast measurement files. Also saves camera files that are compatible with the downsampled datasets. 

## 2. Copy other necessary files to **data** directory
- Copy the necessary json Facet/Facet Ensemble files to the data directory.
- Make sure the *general/* directory exists with necessary files in the *test/data/* direcotory; copy if needed.

## 3. Generate unit test data
Run the following scripts to save unit test data using the downsampled measurement files. Within these files, define names of the input/output files.
- *generate_test_data_multi_facet.py*
- *generate_test_data_single_facet.py*
- *generate_test_data_undefined.py*

## 4. Run Sofast unit tests
Check that the Sofast unit tests run without errors.