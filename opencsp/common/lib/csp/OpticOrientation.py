"""Class to handle relative orientations of heirarchical optics.
"""

from warnings import warn

from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ

warn(
    "OpticOrientation should not be used. Instead, extend an optic object "
    "by OpticOrientationAbstract and implement thae abstract methods. ",
    DeprecationWarning,
)


class OpticOrientation:
    """Object holding orientation parameters relating the following reference frames:
    1) Child mount: This is the platform that a child optic is mounted to. For
       example, a mirror is a child of a Facet.
    2) Base mount: This is the base mount of the object itself.
    3) Parent mount: This is the platform that the optic mounts to its parent
       optic. For example, the parent mount of a mirror is the child mount of a
       Facet.
    """

    def __init__(self, no_child: bool = False, no_parent: bool = False):
        """OpticOrientation is initialized with zero translation and
        zero rotation between all reference frames.

        Parameters
        ----------
        no_child : bool, optional
            Set true if no child mount exists, by default False
        no_parent : bool, optional
            Set true if no parent mount exists, by default False
        """
        self.no_child = no_child
        self.no_parent = no_parent

        if self.no_child:
            self._transform_child_to_base = None
        else:
            self._transform_child_to_base = TransformXYZ.from_zero_zero()

        if self.no_parent:
            self._transform_base_to_parent = None
        else:
            self._transform_base_to_parent = TransformXYZ.from_zero_zero()

    def __repr__(self):
        out = ''
        # Child to base
        if self.no_child:
            out += 'Child to Base: None\n'
        else:
            out += 'Child to Base: ' + str(self.transform_child_to_base.R.as_rotvec().round(3)) + '\n'
        # Base to parent
        if self.no_parent:
            out += 'Base to parent: None\n'
        else:
            out += 'Base to parent: ' + str(self.transform_base_to_parent.R.as_rotvec().round(3)) + '\n'
        # Child to parent
        if self.no_child or self.no_parent:
            out += 'Child to parent: None'
        else:
            out += 'Child to parent: ' + str(self.transform_child_to_base.R.as_rotvec().round(3))
        return out

    @property
    def transform_child_to_base(self) -> TransformXYZ:
        """Transform coordinates from child mount to base mount"""
        if self.no_child:
            raise ValueError('Optic does not have child mount.')
        return self._transform_child_to_base

    @transform_child_to_base.setter
    def transform_child_to_base(self, transform: TransformXYZ) -> None:
        if self.no_child:
            raise ValueError('Optic does not have child mount.')
        if not isinstance(transform, TransformXYZ):
            raise TypeError(f'Input transform must be type {TransformXYZ} but is type {type(transform)}')
        self._transform_child_to_base = transform.copy()

    @property
    def transform_base_to_parent(self) -> TransformXYZ:
        """Transform coordinates from base mount to parent mount"""
        if self.no_parent:
            raise ValueError('Optic does not have parent mount.')
        return self._transform_base_to_parent

    @transform_base_to_parent.setter
    def transform_base_to_parent(self, transform: TransformXYZ) -> TransformXYZ:
        if self.no_parent:
            raise ValueError('Optic does not have parent mount.')
        if not isinstance(transform, TransformXYZ):
            raise TypeError(f'Input transform must be type {TransformXYZ} but is type {type(transform)}')
        self._transform_base_to_parent = transform.copy()

    @property
    def transform_child_to_parent(self) -> TransformXYZ:
        """Transform coordinatges from child mount to parent mount"""
        if self.no_child or self.no_parent:
            raise ValueError('Optic does not have both a child and parent platform.')
        return self._transform_base_to_parent * self._transform_child_to_base
