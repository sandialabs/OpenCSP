"""Example script that connects to and shows a live view from a 12 bit Basler monochrome camera.
"""

import argparse

from opencsp.common.lib.camera.ImageAcquisition_DCAM_mono import ImageAcquisition as ImageAcquisitionMono
from opencsp.common.lib.camera.LiveView import LiveView


def main():
    parser = argparse.ArgumentParser(
        prog='live_view_mono_Basler', description='Shows live view from Basler monochrome camera.'
    )
    parser.add_argument(
        'camera_index',
        metavar='index',
        type=int,
        help='Camera index (0-indexed in order of camera serial number) to run.',
    )
    parser.add_argument('--calibrate', action='store_true', help='calibrate camera exposure before capture')
    parser.add_argument('-e', metavar='exposure', type=int, default=None, help='Camera exposure value')
    parser.add_argument('-g', metavar='gain', type=int, default=None, help='Camera gain value')
    args = parser.parse_args()

    # Connect to camera
    cam = ImageAcquisitionMono(args.camera_index, 'Mono12')

    # Set exposure
    if args.e is not None:
        cam.exposure_time = args.e

    # Set gain
    if args.g is not None:
        cam.gain = args.g

    # Calibrate exposure and set frame rate
    if args.calibrate:
        cam.calibrate_exposure()

    print('Exposure time range: ', cam.cap.ExposureTimeRaw.Min, '-', cam.cap.ExposureTimeRaw.Max)
    print('Gain range: ', cam.cap.GainRaw.Min, '-', cam.cap.GainRaw.Max)

    print('Exposure time set:', cam.exposure_time)
    print('Gain set:', cam.gain)

    # Show live view image
    LiveView(cam, highlight_saturation=False)

    # Close camera once live vieiw is stopped
    cam.close()


if __name__ == '__main__':
    main()
