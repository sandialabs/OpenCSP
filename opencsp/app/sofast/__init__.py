_sofast_server_settings_key = "sofast_server"
_sofast_server_settings_default: dict[str, None] = {
    "log_output_dir": None,
    "saves_output_dir": None,
    "camera_files": None,
    "projector_file": None,
    "calibration_file": None,
    "mirror_measure_point": None,
    "mirror_screen_distance": None,
    "fixed_pattern_diameter_and_spacing": None,
    "spatial_orientation": None,
    "display_shape_file": None,
    "dot_locations_file": None,
    "facet_files": None,
    "ensemble_file": None,
    "surface_shape_file": None,
}
"""
log_output_dir: Where to save log output to from the server.
camera_files: Where to find the camera .h5 file(s), which define the default cameras to connect to on server start.
projector_file: Where to find the projection .h5 file, which defines the default screen space for the projector.
calibration_file: Where to find the calibration .h5 file, which defines the default camera-screen response calibration.
"""

_settings_list = [[_sofast_server_settings_key, _sofast_server_settings_default]]
