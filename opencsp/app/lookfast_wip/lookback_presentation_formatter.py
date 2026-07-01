import opencsp.app.lookback.lookback_tools as lbt
from logging import ERROR, DEBUG
import os

logger = lbt.logging_setup(
    log_folder=os.path.join(os.getcwd, "error_logs"),
    log_file_name="error_log_presentation_formatter.txt",
    log_type=ERROR,
)

# Example usage
if __name__ == "__main__":
    checkpoint_dir = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_0049/0_checkpoints"
    video_file_path = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_0049/DSC_0049.MOV"
    video_file_path_new = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_0049/5_accelerated_test_video/DSC_0049_Accelerated_10x.MOV"

    lbt.accelerate_video_ffmpeg_no_audio(input_path=video_file_path, output_path=video_file_path_new, playback_speed=10)
