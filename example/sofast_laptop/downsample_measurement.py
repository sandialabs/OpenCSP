import contrib.test_data_generation.sofast_fringe.downsample_data as dd
from os.path import join, dirname


def do_downsample():
    dir_measurement_data = join(dirname(__file__), 'data/large_cosmetic_mirror')
    file_calibration_and_measurement = join(dir_measurement_data, '20240515_104737_measurement_fringe.full_res.h5')

    downsampled_data = dd.downsample_measurement(str(file_calibration_and_measurement), 4)

    downsampled_data[0].save_to_hdf("measurement.h5")
    downsampled_data[1].save_to_hdf("calibration.h5")


if __name__ == '__main__':
    do_downsample()
