# SOFAST Contributed Example Suite

Code under this directory is not actively maintained by the OpenCSP team.

This is experimental code, still in early stages of exploration and development.

These examples require additional hardware and data.

The Sofast Contributed example suite contains:
1. Example files that provide examples on how to use Sofast to collect data and process already collected data.
2. Files that can be used to test Sofast's processing on previously processed data using the full-resolution data in the sample data suite.

## Example file description:
| File | Description |
| :--- | :--- |
| run_and_characterize_sofast_1_cam.py | Runs Sofast (requires camera/projector/etc.) with a single camera and characterizes data as a single facet data collection. |
| run_and_characterize_sofast_2_cam.py | Runs Sofast (requires two cameras/projector/etc.) with two cameras simultaneously and characterizes data as a single facet data collection |

## Full-resolution testing with Sofast sample data:
| File | Description |
| :--- | :--- |
| generate_processed_datasets.py | Loads measurement files from the Sofast sample data suite, characterizes with Sofast, and saves output data |
| run_full_calibration_manual.py | Runs a manual Sofast calibration using previously captured calibration data from the Sofast sample data suite |