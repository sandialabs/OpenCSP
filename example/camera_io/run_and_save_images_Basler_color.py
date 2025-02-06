"""
Example program that captures images from a 12 bit color Basler
camera and saves the images in TIFF format.
 
"""

import argparse
import imageio.v3 as imageio

from opencsp.common.lib.camera.ImageAcquisition_DCAM_color import ImageAcquisition as ImageAcquisitionColor


def main():
    parser = argparse.ArgumentParser(
        prog='run_and_save_images_Basler_color',
        description='Captures N frames from a color Basler camera. Saves images as 12bit numbers packed in 16 bit integers in TIFF format with filenames of the form: xx.tiff',
    )
    parser.add_argument(
        'camera_index', type=int, help='Camera index (0-indexed in order of camera serial number) to run.'
    )
    parser.add_argument('num_images', type=int, help='Number of images to capture and save.')
    parser.add_argument('--calibrate', action='store_true', help='Calibrate camera exposure before capture.')
    parser.add_argument(
        '-p',
        '--prefix',
        metavar='prefix',
        default='',
        type=str,
        help='Image save prefix to be appended before the filename.',
    )
    args = parser.parse_args()

    # Connect to camera
    cam = ImageAcquisitionColor(args.camera_index, 'BayerRG12')

    # Calibrate exposure and set frame rate
    cam.frame_rate = 10
    if args.calibrate:
        cam.calibrate_exposure()

    print('Exposure time:', cam.exposure_time)
    print('Frame rate:', cam.frame_rate)
    print('Gain:', cam.gain)
    print('')

    # Capture and save frames
    for idx in range(args.num_images):
        frame = cam.get_frame()
        # Save by packing 12 bit image into 16 bit image
        imageio.imwrite(f'{args.prefix}{idx:02d}.tiff', frame * 2**4)

    # Close camera
    cam.close()


if __name__ == '__main__':
    main()
