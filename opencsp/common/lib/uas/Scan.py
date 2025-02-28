import opencsp.common.lib.uas.ScanPass as sp


class Scan:
    """
    A flight scan, comprised of a sequence of scan passes.

    Attributes
    ----------
    passes : list of ScanPass
        The list of scan passes associated with this flight scan.
    """

    # "ChatGPT 4o" assisted with generating this docstring.

    def __init__(self, scan_passes):  # List of ScanPass objects.
        """
        A flight scan, comprised of a sequence of scan passes.

        Parameters
        ----------
        scan_passes : list of ScanPass
            A list of ScanPass objects representing the individual scan passes that make up the flight scan.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        super(Scan, self).__init__()

        # Check input.
        if len(scan_passes) == 0:
            print("In Scan constructor, empty list of scan passes encountered.")
            assert False

        # Input parameters.
        self.passes = scan_passes

    # ACCESS

    def waypoints(self):
        """
        Retrieve a list of waypoints from all scan passes.

        Returns
        -------
        list
            A list of waypoints collected from each scan pass.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        waypoint_list = []
        for scan_pass in self.passes:
            pass_waypoint_list = scan_pass.waypoints()
            waypoint_list = waypoint_list + pass_waypoint_list
        return waypoint_list

    def locale(self):
        """
        Get the locale of the first scan pass.

        Returns
        -------
        str
            The locale of the first scan pass in the sequence.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return self.passes[0].locale()

    # MODIFICATION

    def set_scan_pass_numbers(self, next_scan_index):
        """
        Set the index for each scan pass and return the next available index.

        Parameters
        ----------
        next_scan_index : int
            The starting index to assign to the scan passes.

        Returns
        -------
        int
            The next available index after assigning numbers to the scan passes.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        for scan_pass in self.passes:
            scan_pass.idx = next_scan_index
            next_scan_index += 1
        return next_scan_index

    # RENDERING

    def draw(self, view, scan_pass_styles):
        """
        Render the scan passes on the specified view using the provided styles.

        Parameters
        ----------
        view : object
            The view object where the scan passes will be rendered.
        scan_pass_styles : dict
            A dictionary containing styles for rendering the scan passes.

        Returns
        -------
        None
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        for scan_pass in self.passes:
            scan_pass.draw(view, scan_pass_styles)


# -------------------------------------------------------------------------------------------------------
# CONSTRUCTION FUNCTIONS
#


def construct_scan_given_segments_of_interest(list_of_xyz_segments, scan_parameters):
    """
    Construct a Scan object given a list of XYZ segments of interest.

    This function creates scan passes based on the provided segments and scan parameters,
    arranging them in a back-and-forth boustrophedon pattern.

    Parameters
    ----------
    list_of_xyz_segments : list of list of float
        A list of segments, where each segment is defined by a list of XYZ coordinates.
    scan_parameters : dict
        A dictionary containing parameters for the scan configuration.

    Returns
    -------
    Scan
        A Scan object constructed from the provided segments and parameters.

    Notes
    -----
    The function assumes that the input segment list is parallel, meaning each segment points
    from p0 to p1 in the same direction. It reverses every other segment to create a boustrophedon pattern.

    Examples
    --------
    >>> scan = construct_scan_given_segments_of_interest([[0, 0, 0], [1, 1, 1]], {'fly_forward_backward': True})
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Notify progress.
    print("Constructing scan given segments of interest...")

    # We assume the input segment list is parallel in the sense that each segment points
    # from p0 --> p1 in the same direction.
    # However, to maximize scanning that can be done within a limited UAS range, we will
    # fly a back-and-forth boustrophedon pattern,* so we reverse every other segment.
    # * Thanks to Laura McNamara for this excellent word!
    #
    list_of_xyz_segments_2 = []
    idx = 0
    for segment_xyz in list_of_xyz_segments:
        if (idx % 2) == 0:
            list_of_xyz_segments_2.append(segment_xyz)
        else:
            reverse_segment_xyz = [segment_xyz[1], segment_xyz[0]]
            list_of_xyz_segments_2.append(reverse_segment_xyz)
        idx += 1
    # Construct scan passes.
    passes = []
    idx = 0
    for segment_xyz in list_of_xyz_segments_2:
        if (not scan_parameters["fly_forward_backward"]) or ((idx % 2) == 0):
            fly_backward = False
        else:
            fly_backward = True
        passes.append(sp.construct_scan_pass_given_segment_of_interest(segment_xyz, fly_backward, scan_parameters))
        idx += 1
    # Construct scan.
    scan = Scan(passes)
    # Return.
    return scan


def construct_scan_given_UFACET_scan_passes(ufacet_scan_pass_list, scan_parameters):
    """
    Construct a Scan object given a list of UFACET scan passes.

    This function creates scan passes based on the provided UFACET scan passes and scan parameters,
    arranging them in a back-and-forth boustrophedon pattern.

    Parameters
    ----------
    ufacet_scan_pass_list : list of ScanPass
        A list of UFACET scan passes to be used for constructing the Scan object.
    scan_parameters : dict
        A dictionary containing parameters for the scan configuration.

    Returns
    -------
    Scan
        A Scan object constructed from the provided UFACET scan passes and parameters.

    Notes
    -----
    The function assumes that UFACET scans are specified by line segments that are parallel,
    and it reverses every other pass to create a boustrophedon pattern.

    Examples
    --------
    >>> scan = construct_scan_given_UFACET_scan_passes(ufacet_passes, {'fly_forward_backward': True})
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Notify progress.
    print("Constructing scan given UFACET scan passes...")

    # UFACET scans are always specified by line segments that are "parallel" in the sense
    # that each segment points from p0 --> p1 in roughly the same direction.
    # However, to maximize scanning that can be done within a limited UAS range, we will
    # fly a back-and-forth boustrophedon pattern,* so we reverse every other segment.
    # * Thanks to Laura McNamara for this excellent word!
    #
    # Because UFACET requires teh camera to face the mirrors during each pass, the
    # UAS flies backward for every other pass.
    #
    # Construct scan passes.
    passes = []
    idx = 0
    for ufacet_scan_pass in ufacet_scan_pass_list:
        if (idx % 2) == 0:
            fly_backward = False
        else:
            fly_backward = True
        passes.append(sp.construct_scan_pass_given_UFACET_scan_pass(ufacet_scan_pass, fly_backward, scan_parameters))
        idx += 1
    # Construct scan.
    scan = Scan(passes)
    # Return.
    return scan
