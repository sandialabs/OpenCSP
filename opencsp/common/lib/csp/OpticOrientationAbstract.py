from abc import abstractmethod
import copy
from warnings import warn

from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ


class OpticOrientationAbstract:
    """
    Classes that extend OpticOrientationAbstract are objects that can be in different
    orientations, and can contain child objects (that also extend OpticOrientationAbstract)
    that transform with it.

    Core Attributes:
    ----------------
    `self._parent`: `OpticOrientationAbstract`
        - This is the `OpticOrientationAbstract` object that contains the self.
        If `self` is the largest object then this attribute will be `None`.
        If an object already has a parent, then it cannot be added to another object as a child.
        This should be accessed via the `@property` method `self.parent`.

    `self._self_to_parent_transform`: `TransformXYZ`
        - This is the relatice transformation from the `self` coordinate frame to
        the `self._parent` frame.

    (`@property`) `self.children`: `list[OpticOrientationAbstract]`
        - This property is an abstract method that must be implemented by all deriving classes.
        There should not be a setter for this atribute. Ideally, there is little information that
        is stored in two places. However, for both parents and children to find eachother,
        the information about children must be stored in the parent. When implementing `self.children`,
        ensure that children hold reference to parents.

    Derived Attributes
    ------------------
    Other attributes of a `OpticOrientationAbstract` are caluculated based on the other information
    in the core attributes.

    - `_children_to_self_transform`
    - `self_to_global_tranformation`
    - `get_transform_relative_to`
    - `get_most_basic_optics`

    This minimized what information must be stored in multiple locations.

    """

    def __init__(self) -> None:
        self._parent: 'OpticOrientationAbstract' = None
        self._self_to_parent_transform: TransformXYZ = None

        self._set_optic_children()

    def no_parent_copy(self):
        """Deep copy of Optic without a parential attachement."""
        copy_of_self = copy.deepcopy(self)
        copy_of_self._parent = None
        return copy_of_self

    def _set_optic_children(self) -> None:
        # Call this function in the constructor to set the children of self
        # and set self as the parent of the children.
        if self.children is not None:
            for child in self.children:
                # print("debug child", child, "debug self", self)
                child._parent = self
                child._self_to_parent_transform = TransformXYZ.identity()  # defult to no transformation

    @property
    @abstractmethod
    def children(self) -> list['OpticOrientationAbstract']:
        """Returns the children of this instance of OpticOrientationAbstract decendent."""
        raise NotImplementedError("abstract property children must be overwritten.")

    @abstractmethod
    def _add_child_helper(self, new_child: 'OpticOrientationAbstract'):
        "Add child OpticOrientationAbstract object to self."

    def add_child(self, new_child: 'OpticOrientationAbstract', new_child_to_self_transform=TransformXYZ.identity()):
        """Adds a child to the current optic"""
        if new_child.parent is not None:
            raise ValueError(
                "Cannot add a child optic if that child is already owned by a parent: \n"
                f"new_child: {new_child} \n"
                f"new_child.parent: {new_child.parent} \n"
                f"Intended partent: {self}"
            )
        self._add_child_helper(new_child)
        new_child._parent = self
        new_child._self_to_parent_transform = new_child_to_self_transform

    ## TODO: If this `remove_child` going to be added it needs to be done in a way that does not break
    ## the expected behavior of an optic. For example, will this be the method that is
    ## supposed to be used to remove a heliostat from a field?
    # def remove_child(self, child_to_remove: 'OpticOrientationAbstract'):
    #     if child_to_remove.parent is not self:
    #         raise ValueError("Cannot remove a child optic if that child is owned by specified parent: \n"
    #                          f"child_to_remove: {child_to_remove} \n"
    #                          f"child_to_remove.parent: {child_to_remove.parent} \n"
    #                          f"Intended partent: {self}")
    #     child_to_remove._parent = None

    @property
    def parent(self) -> 'OpticOrientationAbstract':
        """The parent of the current Optic"""
        return self._parent

    @property
    def _children_to_self_transform(self) -> list[TransformXYZ]:
        # The list of transformations that take each child OpticOrientationAbstract object
        # from their frames of reference into the 'self' frame.
        if self.children is None:
            return None
        return [child._self_to_parent_transform for child in self.children]

    @property
    def self_to_global_tranformation(self) -> TransformXYZ:
        """
        Gets the transformation from '`self`' frame to the global frame
        where global is the object in the ancestor tree that does not have a parent.

        Parameters
        ----------
        `self`: `OpticOrientationAbstract`
            An object that derives OpticOrientationAbstract

        Returns
        -------
        `TransformXYZ`
            The transformation that takes self to the global frame of reference
        """
        searcher = self
        transform = TransformXYZ.identity()
        while searcher._parent is not None:
            transform = searcher._self_to_parent_transform * transform
            searcher = searcher._parent
        return transform

    def get_transform_relative_to(self: 'OpticOrientationAbstract', target: 'OpticOrientationAbstract') -> TransformXYZ:
        """
        Gets the transformation from '`self`' frame to the '`target`' frame if `target`
        is an optic ancestor or decendent of `self`.

        Parameters
        ----------
        `self`: `OpticOrientationAbstract`
            An object that derives OpticOrientationAbstract
        `target`: `OpticOrientationAbstract`
            An object that derives OpticOrientationAbstract

        Returns
        -------
        `TransformXYZ`
            The transformation that takes self to the `target` frame of reference
        """
        if target is self:
            return TransformXYZ.identity()

        # look up the tree
        searcher = self
        transform = TransformXYZ.identity()
        while searcher.parent is not None:
            transform = searcher._self_to_parent_transform * transform
            searcher = searcher._parent
            if searcher is target:
                return transform

        # look down the tree
        searcher = target
        transform = TransformXYZ.identity()
        while searcher.parent is not None:
            transform = searcher._self_to_parent_transform * transform
            searcher = searcher._parent
            if searcher is self:
                return transform.inv()

        raise ValueError("The given 'target' is not an in the current parent-child tree.")

    def get_most_basic_optics(self) -> list['OpticOrientationAbstract']:
        """Return the list of the smallest optic that makes up composite optics."""
        warn("Funciton is not verified")
        if self.children is None:
            return [self]

        basic_optics: list['OpticOrientationAbstract'] = []
        for child in self.children:
            basic_optics += child.get_most_basic_optics()
        return basic_optics
