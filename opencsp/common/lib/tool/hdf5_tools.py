from abc import abstractmethod, ABC
import os

import h5py
import numpy as np

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


def save_hdf5_datasets(data: list, datasets: list, file: str):
    """Saves data to HDF5 file"""
    with h5py.File(file, "a") as f:
        # Loop through datasets
        for d, dataset in zip(data, datasets):
            if dataset in f:
                # Delete dataset if it already exists
                del f[dataset]
            # Write dataset
            f.create_dataset(dataset, data=d)


def load_hdf5_datasets(datasets: list, file: str):
    """Loads datasets from HDF5 file"""
    with h5py.File(file, "r") as f:
        kwargs: dict[str, str | h5py.Dataset] = {}
        # Loop through fields to retreive
        for dataset in datasets:
            # Get data and get dataset name
            data = f[dataset]
            name = dataset.split("/")[-1]

            # Format data shape
            if np.ndim(data) == 0 and np.size(data) == 1:
                data = data[()]
            elif np.ndim(data) > 0 and np.size(data) == 1:
                data = data[...].flatten()[0]
            elif np.size(data) > 0:
                data = data[...].squeeze()

            # Format strings
            if type(data) is np.bytes_ or type(data) is bytes:
                data = data.decode()

            # Save in dictionary
            kwargs.update({name: data})

    return kwargs


def is_dataset_and_shape(object: h5py.Group | h5py.Dataset) -> tuple[bool, tuple]:
    """Returns whether the given object is an hdf5 dataset and, if it is, then
    also what it's shape is.

    Parameters
    ----------
    object : h5py.Group | h5py.Dataset
        The object to check the type of.

    Returns
    -------
    is_dataset: bool
        True if object is a dataset, False otherwise
    shape: tuple[int]
        The shape of the dataset. Empty tuple() object if not a dataset.
    """
    if not isinstance(object, h5py.Group):
        if isinstance(object, h5py.Dataset):
            dset: h5py.Dataset = object
            return True, dset.shape
        else:
            return True, tuple()
    else:
        return False, tuple()


def get_groups_and_datasets(hdf5_path_name_ext: str | h5py.File):
    """Get the structure of an HDF5 file, including all group and dataset names, and the dataset shapes.

    Parameters
    ----------
    hdf5_path_name_ext : str | h5py.File
        The HDF5 file to parse the structure of.

    Returns
    -------
    group_names: list[str]
        The absolute names of all the groups in the file. For example: "foo/bar"
    file_names_and_shapes: list[ tuple[str,tuple[int]] ]
        The absolute names of all the datasets in the file, and their shapes.
        For example: "foo/bar/baz", (1920,1080)
    """
    group_names: list[str] = []
    file_names_and_shapes: list[tuple[str, tuple[int]]] = []
    visited: list[tuple[str, bool, tuple]] = []

    def visitor(name: str, object: h5py.Group | h5py.Dataset):
        visited.append(tuple([name, *is_dataset_and_shape(object)]))
        return None

    if isinstance(hdf5_path_name_ext, str):
        hdf5_path_name_ext = ft.norm_path(hdf5_path_name_ext)
        with h5py.File(hdf5_path_name_ext, "r") as fin:
            fin.visititems(visitor)
    else:
        fin: h5py.File = hdf5_path_name_ext
        fin.visititems(visitor)

    for name, is_dataset, shape in visited:
        # Add to the file or group names list.
        # If a dataset, then include its shape.
        if not is_dataset:
            group_names.append(name)
        if is_dataset:
            file_names_and_shapes.append(tuple([name, shape]))

    return group_names, file_names_and_shapes


def _create_dataset_path(base_dir: str, h5_dataset_path_name: str, dataset_ext: str = ".txt"):
    dataset_location, dataset_name, _ = ft.path_components(h5_dataset_path_name)
    dataset_path = ft.norm_path(os.path.join(base_dir, dataset_location))
    ft.create_directories_if_necessary(dataset_path)
    return ft.norm_path(os.path.join(dataset_path, dataset_name + dataset_ext))


def unzip(hdf5_path_name_ext: str, destination_dir: str, dataset_format="npy"):
    """Unpacks the given HDF5 file into the given destination directory.

    Unpacks the given HDF5 file into the given destination directory. A new
    directory is created in the destination with the same name as the hdf5 file.
    String values are extracted as .txt files, and images are extracted as .png
    files. Everything else is saved with numpy as .npy files.

    Parameters
    ----------
    hdf5_path_name_ext : str
        The HDF5 file to unpack.
    destination_dir : str
        The directory in which to create a directory for the HDF5 file.
    dataset_format : str, optional
        Format in which to save non-image, non-string datasets. Can be one of
        "npy" or "csv". If the dataset has more than 2 dimension then it will be
        forceably saved to npy. Default npy.

    Returns
    -------
    output_dir: str
        The path to the newly created directory into which the HDF5 files were
        extracted into.
    """
    norm_path = ft.norm_path(hdf5_path_name_ext)
    path, name, ext = ft.path_components(norm_path)
    hdf5_dir = ft.norm_path(os.path.join(destination_dir, name))

    # Create the HDF5 output directory
    if ft.directory_exists(hdf5_dir):
        lt.error_and_raise(FileExistsError, f"Error in hdf5_tools.unzip(): output directory {hdf5_dir} already exists!")
    ft.create_directories_if_necessary(hdf5_dir)

    # Get all of what may be strings or images from the h5 file
    _, dataset_names_and_shapes = get_groups_and_datasets(norm_path)
    possible_strings: list[tuple[str, tuple[int]]] = []
    possible_images: list[tuple[str, tuple[int]]] = []
    other_datasets: list[tuple[str, tuple[int]]] = []
    for dataset_name, shape in dataset_names_and_shapes:
        possible_strings.append(tuple([dataset_name, shape]))

    # Extract strings into .txt files
    possible_strings_names = [t[0] for t in possible_strings]
    for i, possible_string_name in enumerate(possible_strings_names):
        dataset_name = possible_string_name.split("/")[-1]
        h5_val = load_hdf5_datasets([possible_string_name], norm_path)[dataset_name]
        if isinstance(h5_val, np.ndarray) and h5_val.ndim <= 1 and isinstance(h5_val.tolist()[0], str):
            h5_val = h5_val.tolist()[0]
        if isinstance(h5_val, str):
            dataset_path_name_ext = _create_dataset_path(hdf5_dir, possible_strings[i][0], ".txt")
            with open(dataset_path_name_ext, "w") as fout:
                fout.write(h5_val)
        else:
            possible_images.append(possible_strings[i])

    # Extract images into .png files
    possible_images_names = [t[0] for t in possible_images]
    for i, possible_image_name in enumerate(possible_images_names):
        dataset_name = possible_image_name.split("/")[-1]
        h5_val = load_hdf5_datasets([possible_image_name], norm_path)[dataset_name]
        shape = possible_images[i][1]
        if isinstance(h5_val, (h5py.Dataset, np.ndarray)):
            np_image = np.array(h5_val).squeeze()

            # we assume images have 2 or 3 dimensions
            if (len(shape) == 2) or (len(shape) == 3):
                # we assume shapes are at least 10x10 pixels and have an aspect ratio of at least 10:1
                aspect_ratio = max(shape[0], shape[1]) / min(shape[0], shape[1])
                if (shape[0] >= 10 and shape[1] >= 10) and (aspect_ratio < 10.001):
                    dataset_path_name_ext = _create_dataset_path(hdf5_dir, possible_images[i][0], ".png")
                    # assumed grayscale or RGB
                    if (len(shape) == 2) or (shape[2] in [1, 3]):
                        img = it.numpy_to_image(np_image)
                        img.save(dataset_path_name_ext)
                    else:  # assumed multiple images
                        dp, dn, de = ft.path_components(dataset_path_name_ext)
                        for i in range(shape[2]):
                            dataset_path_name_ext_i = os.path.join(dp, f"{dn}_{i}{de}")
                            np_single_image = np_image[:, :, i].squeeze()
                            img = it.numpy_to_image(np_single_image)
                            img.save(dataset_path_name_ext_i)
                else:
                    other_datasets.append(possible_images[i])
            else:
                other_datasets.append(possible_images[i])
        else:
            other_datasets.append(possible_images[i])

    # Extract everything else into numpy or csv arrays
    other_dataset_names = [t[0] for t in other_datasets]
    for i, other_dataset_name in enumerate(other_dataset_names):
        dataset_name = other_dataset_name.split("/")[-1]
        h5_val = load_hdf5_datasets([other_dataset_name], norm_path)[dataset_name]
        np_val = np.array(h5_val)
        dataset_path_name = _create_dataset_path(hdf5_dir, other_datasets[i][0], "")

        if dataset_format == "npy":
            # save as a numpy file
            np.save(dataset_path_name, np_val, allow_pickle=False)
        elif dataset_format == "csv":
            # save as a csv file
            squeezed = np_val.squeeze()
            if len(squeezed.shape) == 0:
                # reshape 0d arrays to be 1d
                squeezed = squeezed.reshape((1))
            if len(squeezed.shape) >= 3:
                # reshape 3d arrays to be 2d for writing to csvs
                longest_dim = max(squeezed.shape)
                squeezed_1d = np.ravel(squeezed)
                other_dim = int(squeezed_1d.size / longest_dim)
                squeezed_2d = np.reshape(squeezed, (longest_dim, other_dim))
                squeezed = squeezed_2d
            np.savetxt(dataset_path_name + ".csv", squeezed, delimiter=",")

    return hdf5_dir


class HDF5_SaveAbstract(ABC):
    """Abstract class for saving to HDF5 format"""

    @abstractmethod
    def save_to_hdf(self, file: str, prefix: str = "") -> None:
        """Saves data to given file. Data is stored as: PREFIX + Folder/Field_1

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """


class HDF5_IO_Abstract(HDF5_SaveAbstract):
    """Abstract class for loading from HDF5 format"""

    @classmethod
    @abstractmethod
    def load_from_hdf(cls, file: str, prefix: str = ""):
        """Loads data from given file. Assumes data is stored as: PREFIX + Folder/Field_1

        Parameters
        ----------
        file : str
            HDF file to load from
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """
