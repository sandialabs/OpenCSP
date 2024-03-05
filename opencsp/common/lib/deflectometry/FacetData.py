"""Data class to store single facet optic definition
"""
import json

from opencsp.common.lib.geometry.Vxyz import Vxyz


class FacetData:
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

    def copy(self) -> 'FacetData':
        """Returns copy of class"""
        return FacetData(
            self.v_facet_corners.copy(),
            self.v_facet_centroid.copy(),
        )

    @classmethod
    def load_from_json(cls, file: str) -> 'FacetData':
        """
        Loads facet definition data from JSON file.

        Parameters
        ----------
        file : str
            JSON file to load.

        """

        # Read JSON
        with open(file, 'r', encoding='utf-8') as f:
            data_json = json.load(f)

        # Put data in class
        return cls(
            v_facet_corners=_Vxyz_from_dict(data_json['v_facet_corners']),
            v_facet_centroid=_Vxyz_from_dict(data_json['v_centroid_facet']),
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
            'v_facet_corners': _Vxyz_to_dict(self.v_facet_corners),
            'v_centroid_facet': _Vxyz_to_dict(self.v_facet_centroid),
        }

        # Save data in JSON
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=3)


def _Vxyz_to_dict(V: Vxyz) -> dict:
    d = {
        'x': V.x.tolist(),
        'y': V.y.tolist(),
        'z': V.z.tolist(),
    }
    return d


def _Vxyz_from_dict(d: dict) -> Vxyz:
    return Vxyz((d['x'], d['y'], d['z']))
