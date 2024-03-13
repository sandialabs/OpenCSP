# Fixed Pattern (FP) Deflectometry
This tool is used to measure the surface slope of mirrors with one image using a fixed pattern. 

## Setup
- **FixedPatternSetupCalibrate.py** - Class used to measure the xyz location of monochrome dots. Typically used for printed/physical patterns, not projected patterns. A SceneReconstruction calibration to locate Aruco markers must be performed first.
- **DotLocationsFixedPattern.py** - Class storing the xyz locations of monochrome dots for a fixed pattern deflectometry measurement. Can be created by measuring the dot locations, or by analyzing a previously calibrate Display object.

## Measurement
- **FixedPatternScreenProjection.py** - Class for creating a monochrome dot pattern to display on a screen. This would be used instead of a physical/printed dot pattern.
- **MeasurementFixedPattern.py** - Class holding all data taken during a fixed pattern deflectometry measurement. Can be used for either projected or physical/printed targets.

## Processing
- **FixedPatternProcess.py** - Class for processing monochrome dot fixed pattern deflectometry.
- **FixedPatternProcessParams.py** - Class holding processing parameters used in processing fixed pattern deflectometry calculations.
