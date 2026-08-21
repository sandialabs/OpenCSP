# Core modules
from . import analysis, utils

# Configuration classes (Phase 3)
from .config import (
    AnalysisConfig,
    TimeConfig,
    LocationConfig,
    FieldConfig,
    HeliostatConfig,
    AimConfig,
    VoxelConfig,
    OpticalConfig,
)

# Results classes (Phase 3)
from .results import (
    AnalysisResults,
    HeliostatResults,
    IrradianceResults,
    ThresholdResults,
)

# Output classes (Phase 3)
from .output import AnalysisOutput

# Legacy class (still primary interface)
from .analysis_data import AnalysisData

# C++ interface (Phase 2)
from .cpp_interface import FFIAMLibrary, AnalysisParams, RawResults