"""
Example script that connects to and shows a live view from an
8 bit Basler monochrome camera.

"""

import argparse

from opencsp.common.lib.camera.ImageAcquisition_DCAM_mono import (
    ImageAcquisition as ImageAcquisitionMono,
)
from opencsp.common.lib.camera.LiveView import LiveView


def main():
    parser = argparse.ArgumentParser(
        prog='run_and_save_images_Basler_color',
        description='Shows live view from Basler monochrome camera.',
    )
    parser.add_argument(
        'camera_index',
        metavar='index',
        type=int,
        help='Camera index (0-indexed in order of camera serial number) to run.',
    )
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='calibrate camera exposure before capture',
    )
    parser.add_argument(
        '-e', metavar='exposure', type=float, default=None, help='Camera exposure value'
    )
    args = parser.parse_args()

    # Connect to camera
    cam = ImageAcquisitionMono(args.camera_index, 'Mono12')

    # Calibrate exposure and set frame rate
    cam.frame_rate = 10
    if args.calibrate:
        cam.calibrate_exposure()

    if args.e is not None:
        cam.exposure_time = args.e

    print('Exposure time:', cam.exposure_time)
    print('Frame rate:', cam.frame_rate)
    print('Gain:', cam.gain)
    print('')

    # Show live view image
    LiveView(cam, highlight_saturation=False)

    # Close camera once live vieiw is stopped
    cam.close()


if __name__ == '__main__':
    main()
