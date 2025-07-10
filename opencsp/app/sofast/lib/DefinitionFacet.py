"""Data class to store single facet optic definition"""

import json

from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool import hdf5_tools


class DefinitionFacet:
    """Single facet optic definition for Sofast"""

    def __init__(self, v_facet_corners: Vxyz, v_facet_centroid: Vxyz):
        """
        Facet Data Definitions
        ----------------------
        v_facet_corners : Vxyz
            Corners of facet in facet coordinates
        v_centroid_facet : Vxyz
            Centroid of facet in facet coordinates

        NOTE: "facet" coordinates are defined as +x to right and +y up when
        looking at the reflective surface of the mirror.

        """
        self.v_facet_corners = v_facet_corners
        self.v_facet_centroid = v_facet_centroid

    def copy(self) -> "DefinitionFacet":
        """Returns copy of class"""
        return DefinitionFacet(self.v_facet_corners.copy(), self.v_facet_centroid.copy())

    @classmethod
    def load_from_json(cls, file: str) -> "DefinitionFacet":
        """
        Loads facet definition data from JSON file.

        Parameters
        ----------
        file : str
            JSON file to load.

        """

        # Read JSON
        with open(file, "r", encoding="utf-8") as f:
            data_json = json.load(f)

        # Put data in class
        return cls(
            v_facet_corners=_Vxyz_from_dict(data_json["v_facet_corners"]),
            v_facet_centroid=_Vxyz_from_dict(data_json["v_centroid_facet"]),
        )

    def save_to_json(self, file: str) -> None:
        """
        Saves facet definition data to JSON format.

        Parameters
        ----------
        file : str
            JSON file to save data into.

        """
        # Save data in dictionary
        data_dict = {
            "v_facet_corners": _Vxyz_to_dict(self.v_facet_corners),
            "v_centroid_facet": _Vxyz_to_dict(self.v_facet_centroid),
        }

        # Save data in JSON
        with open(file, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=3)

    def save_to_hdf(self, file: str, prefix: str = "") -> None:
        """Saves data to given HDF5 file. Data is stored in PREFIX + DefinitionFacet/...

        Parameters
        ----------
        file : str
            HDF filename
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        data = [self.v_facet_corners.data, self.v_facet_centroid.data]
        datasets = [prefix + "DefinitionFacet/v_facet_corners", prefix + "DefinitionFacet/v_facet_centroid"]
        hdf5_tools.save_hdf5_datasets(data, datasets, file)

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str) -> "DefinitionFacet":
        """Loads DefinitionFacet object from given file.  Data is stored in PREFIX + DefinitionFacet/...

        Parameters
        ----------
        file : str
            HDF filename
        prefix : str
            Prefix appended to folder path within HDF file (folders must be separated by "/")
        """
        datasets = [prefix + "DefinitionFacet/v_facet_corners", prefix + "DefinitionFacet/v_facet_centroid"]
        data = hdf5_tools.load_hdf5_datasets(datasets, file)
        v_facet_corners = Vxyz(data["v_facet_corners"])
        v_facet_centroid = Vxyz(data["v_facet_centroid"])
        return cls(v_facet_corners, v_facet_centroid)


def _Vxyz_to_dict(V: Vxyz) -> dict:
    d = {"x": V.x.tolist(), "y": V.y.tolist(), "z": V.z.tolist()}
    return d


def _Vxyz_from_dict(d: dict) -> Vxyz:
    return Vxyz((d["x"], d["y"], d["z"]))
