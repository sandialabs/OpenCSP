"""Data class to store facet ensemble optic definition for sofast
"""

from copy import deepcopy
import json

import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool import hdf5_tools


class DefinitionEnsemble:
    """Facet Ensemble definition for Sofast"""

    def __init__(
        self,
        v_facet_locations: Vxyz,
        r_facet_ensemble: list[Rotation],
        ensemble_perimeter: np.ndarray,
        v_centroid_ensemble: Vxyz,
    ) -> 'DefinitionEnsemble':
        """Creates Facet Ensemble object from data

        Optic Data Definitions
        ----------------------
        v_facet_locations : Vxyz
            Locations of facet origins in ensemble coordinates
        r_facet_ensemble : list[Rotation]
            Rotation to convert ensemble to facet coordinates
        ensemble_perimeter : list[tuple[int, int], ...]
            List of facet/corner indices that define the overall perimeter of
            the mirror ensemble. [(facet_index, facet_corner_index), ...]
        v_centroid_ensemble : Vxyz
            Location of ensemble centroid in ensemble coordinates

        NOTE: "Ensemble" coordinates are defined as +x to right and +y up when
        looking at the reflective surface of the mirrors.

        """
        self.v_facet_locations = v_facet_locations
        self.r_facet_ensemble = r_facet_ensemble
        self.ensemble_perimeter = ensemble_perimeter
        self.v_centroid_ensemble = v_centroid_ensemble

        if len(v_facet_locations) != len(r_facet_ensemble):
            raise ValueError(
                f'Number of facet locations, {len(v_facet_locations):d}, does not match number of facet rotations, {len(r_facet_ensemble):d}.'
            )

        self.num_facets = len(r_facet_ensemble)

    def __copy__(self):
        return self.copy()

    def copy(self) -> 'DefinitionEnsemble':
        """Returns copy of ensemble data"""
        return DefinitionEnsemble(
            self.v_facet_locations.copy(),
            [deepcopy(rot) for rot in self.r_facet_ensemble],
            self.ensemble_perimeter.copy(),
            self.v_centroid_ensemble.copy(),
        )

    @classmethod
    def load_from_json(cls, file: str) -> 'DefinitionEnsemble':
        """
        Loads facet ensemble definition data from JSON file.

        Parameters
        ----------
        file : str
            JSON file to load.

        """
        # Read JSON
        with open(file, 'r', encoding='utf-8') as f:
            data_json = json.load(f)

        ensemble_perimeter = np.array(
            (data_json['ensemble_perimeter']['facet_indices'], data_json['ensemble_perimeter']['corner_indices'])
        ).T  # Nx2 ndarray

        # Put data in dictionary
        return cls(
            v_facet_locations=_Vxyz_from_dict(data_json['v_facet_locations']),
            r_facet_ensemble=_rot_list_from_dict(data_json['r_facet_ensemble']),
            ensemble_perimeter=ensemble_perimeter,
            v_centroid_ensemble=_Vxyz_from_dict(data_json['v_centroid_ensemble']),
        )

    def save_to_json(self, file: str) -> None:
        """
        Saves facet ensemble definition data to JSON format.

        Parameters
        ----------
        file : str
            JSON file to save data into.

        """
        ensemble_perimeter = self.ensemble_perimeter

        data_dict = {
            'v_facet_locations': _Vxyz_to_dict(self.v_facet_locations),  # Vxyz
            'r_facet_ensemble': _rot_list_to_dict(self.r_facet_ensemble),  # list[Rotation]
            'ensemble_perimeter': {
                'facet_indices': ensemble_perimeter[:, 0].tolist(),  # list
                'corner_indices': ensemble_perimeter[:, 1].tolist(),  # list
            },
            'v_centroid_ensemble': _Vxyz_to_dict(self.v_centroid_ensemble),  # Vxyz
        }

        # Save data in JSON
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=3)

    def save_to_hdf(self, file: str, prefix: str = '') -> None:
        """Saves data to given HDF5 file. Data is stored in PREFIX + DefinitionEnsemble/...

        Parameters
        ----------
        file : str
            HDF filename
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        data = [
            self.v_facet_locations.data,
            np.array([r.as_rotvec() for r in self.r_facet_ensemble]),  # Nx3 vector
            self.ensemble_perimeter,
            self.v_centroid_ensemble.data,
        ]
        datasets = [
            prefix + 'DefinitionEnsemble/v_facet_locations',
            prefix + 'DefinitionEnsemble/r_facet_ensemble',
            prefix + 'DefinitionEnsemble/ensemble_perimeter',
            prefix + 'DefinitionEnsemble/v_centroid_ensemble',
        ]
        hdf5_tools.save_hdf5_datasets(data, datasets, file)

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str = '') -> 'DefinitionEnsemble':
        """Loads DefinitionEnsemble object from given file.  Data is stored in PREFIX + DefinitionEnsemble/...

        Parameters
        ----------
        file : str
            HDF filename
        prefix : str
            Prefix appended to folder path within HDF file (folders must be separated by "/")
        """
        datasets = [
            prefix + 'DefinitionEnsemble/v_facet_locations',
            prefix + 'DefinitionEnsemble/r_facet_ensemble',
            prefix + 'DefinitionEnsemble/ensemble_perimeter',
            prefix + 'DefinitionEnsemble/v_centroid_ensemble',
        ]
        data = hdf5_tools.load_hdf5_datasets(datasets, file)
        v_facet_locations = Vxyz(data['v_facet_locations'])
        r_facet_ensemble = [Rotation.from_rotvec(r) for r in data['r_facet_ensemble']]
        ensemble_perimeter = data['ensemble_perimeter']
        v_centroid_ensemble = Vxyz(data['v_centroid_ensemble'])
        return cls(v_facet_locations, r_facet_ensemble, ensemble_perimeter, v_centroid_ensemble)


def _Vxyz_to_dict(V: Vxyz) -> dict:
    d = {'x': V.x.tolist(), 'y': V.y.tolist(), 'z': V.z.tolist()}
    return d


def _Vxyz_from_dict(d: dict) -> Vxyz:
    return Vxyz((d['x'], d['y'], d['z']))


def _rot_list_to_dict(rot: list[Rotation]) -> dict:
    d = {}
    for idx, r in enumerate(rot):
        r1 = r.as_rotvec().squeeze().tolist()
        d.update({str(idx): r1})
    return d


def _rot_list_from_dict(d: dict) -> list[Rotation]:
    return [Rotation.from_rotvec(v) for v in d.values()]
