"""


"""

import math
import numpy as np

import opencsp.common.lib.uas.WayPoint as wp


class ScanPass:
    """
    Model a scan pass within a flight.

    This class represents a single scan pass, which consists of waypoints and associated parameters
    for a flight operation.

    Attributes
    ----------
    idx : int
        The index of the scan pass.
    """

    # "ChatGPT 4o" assisted with generating this docstring.

    def __init__(self):
        """
        Model a scan pass within a flight.

        This class represents a single scan pass, which consists of waypoints and associated parameters
        for a flight operation.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        super(ScanPass, self).__init__()

        # Constructed members.
        self.idx = -1  # Integer.  Not set yet.
        self._core_waypoint_list = None  # Do not access this member externally; use waypoints() function instead.
        self._waypoint_list = None  # Do not access this member externally; use waypoints() function instead.

        # Defining members.
        self._segment_dict = None  # Dictionary of defining parameters.  Only set if this is a segment scan.
        self._ufacet_scan_pass = None  # A UfacetScanPass object.  Only set if this is a UFACET scan.
        self._lead_in = None  # m. Do not access this member externally; use lead_in() function instead.
        self._run_past = None  # m. Do not access this member externally; use run_past() function instead.
        self._locale = None  # Do not access this member externally; use locale() function instead.

    # ACCESS

    def core_waypoints(self):
        """
        Retrieve the core waypoints for the scan pass.

        Returns
        -------
        list
            A list of core waypoints for the scan pass.

        Raises
        ------
        AssertionError
            If the core waypoint list is not set.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # Core waypoints do not include the lead-in and run-past extra travel distance.
        if self._core_waypoint_list == None:
            print("ERROR: In ScanPass.core_waypoints(), attempt to fetch unset _core_waypoint_list.")
            assert False
        return self._core_waypoint_list

    def waypoints(self):
        """
        Retrieve all waypoints for the scan pass.

        Returns
        -------
        list
            A list of all waypoints for the scan pass.

        Raises
        ------
        AssertionError
            If the waypoint list is not set.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        if self._waypoint_list == None:
            print("ERROR: In ScanPass.waypoints(), attempt to fetch unset _waypoint_list.")
            assert False
        return self._waypoint_list

    def segment_dict(self):
        """
        Retrieve the segment dictionary for the scan pass.

        Returns
        -------
        dict
            The segment dictionary for the scan pass.

        Raises
        ------
        AssertionError
            If the segment dictionary is not set.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        if self._segment_dict == None:
            print("In ScanPass.segment_dict(), attempt to fetch unset _segment_dict.")
            assert False
        return self._segment_dict

    def ufacet_scan_pass(self):
        """
        Retrieve the associated UFACET scan pass.

        Returns
        -------
        UfacetScanPass
            The associated UFACET scan pass.

        Raises
        ------
        AssertionError
            If the UFACET scan pass is not set.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        if self._ufacet_scan_pass == None:
            print("In ScanPass.ufacet_scan_pass(), attempt to fetch unset _ufacet_scan_pass.")
            assert False
        return self._ufacet_scan_pass

    def lead_in(self):
        """
        Retrieve the lead-in distance for the scan pass.

        Returns
        -------
        float
            The lead-in distance for the scan pass.

        Raises
        ------
        AssertionError
            If the lead-in distance is not set.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        if self._lead_in == None:
            print("In ScanPass.lead_in(), attempt to fetch unset _lead_in.")
            assert False
        return self._lead_in

    def run_past(self):
        """
        Retrieve the run-past distance for the scan pass.

        Returns
        -------
        float
            The run-past distance for the scan pass.

        Raises
        ------
        AssertionError
            If the run-past distance is not set.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        if self._run_past == None:
            print("In ScanPass.run_past(), attempt to fetch unset _run_past.")
            assert False
        return self._run_past

    def locale(self):
        """
        Retrieve the locale for the scan pass.

        Returns
        -------
        str
            The locale for the scan pass.

        Raises
        ------
        AssertionError
            If the locale is not set.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # Defined bor both raster and UFACET scans, but the input source differs.
        # Cache here to provide uniform access.
        if self._locale == None:
            print("ERROR: In ScanPass.locale(), attempt to fetch unset _locale.")
            assert False
        return self._locale

    # MODIFICATION

    def set_core_waypoints_given_segment_of_interest(self, segment_xyz, fly_backward, raster_scan_parameters):
        """
        Set the core waypoints based on a segment of interest.

        Parameters
        ----------
        segment_xyz : list of list of float
            A list of XYZ coordinates defining the segment of interest.
        fly_backward : bool
            A flag indicating whether to fly backward along the segment.
        raster_scan_parameters : dict
            A dictionary containing parameters for the raster scan.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the gaze angle is positive.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # This routine does not consider ground clearance.
        # It assumes that the specified eta and relative_z will yield a collisiion-free result.
        #
        # Check input.
        if raster_scan_parameters["eta"] > 0:
            print("ERROR: In ScanPass.set_waypoints_given_segment_of_interest(), Postive gaze angle encountered.")
            assert False
        # Set segment-specific data member.
        self._segment_dict = {}
        self._segment_dict["segment_xyz"] = segment_xyz
        # Fetch segment scan parameters.
        locale = raster_scan_parameters["locale"]
        relative_z = raster_scan_parameters["relative_z"]
        eta = raster_scan_parameters["eta"]
        speed = raster_scan_parameters["speed"]
        # Save locale.
        self._locale = locale
        # Fetch segment coords.
        xyz0 = segment_xyz[0]
        x0 = xyz0[0]
        y0 = xyz0[1]
        z0 = xyz0[2]
        xyz1 = segment_xyz[1]
        x1 = xyz1[0]
        y1 = xyz1[1]
        z1 = xyz1[2]
        # Heading is direction of segment.
        dx = x1 - x0
        dy = y1 - y0
        theta = math.atan2(dy, dx)
        if fly_backward:
            theta += np.pi
            if theta < (2 * np.pi):
                theta += 2 * np.pi
            if theta > (2 * np.pi):
                theta -= 2 * np.pi
        # Construct lateral offset.
        d_offset = relative_z / np.tan(-eta)
        x_offset = d_offset * np.cos(theta)
        y_offset = d_offset * np.sin(theta)
        # Vertical offset is due to slope in segment.
        dz = z1 - z0
        d = np.sqrt((dx * dx) + (dy * dy))
        sigma = math.atan2(dz, d)  # "sigma" for "slope'
        z_offset = d_offset * np.sin(sigma)
        # Construct start and end points.
        x_start = x0 - x_offset
        y_start = y0 - y_offset
        z_start = z0 + relative_z - z_offset
        start_xyz = [x_start, y_start, z_start]
        x_end = x1 - x_offset
        y_end = y1 - y_offset
        z_end = z1 + relative_z - z_offset
        end_xyz = [x_end, y_end, z_end]
        # Assume constant gaze angle.
        start_eta = eta
        end_eta = eta
        # Construct way points.
        start_wpt = wp.WayPoint(locale, start_xyz, theta, start_eta, stop=False, speed=speed)
        end_wpt = wp.WayPoint(locale, end_xyz, theta, end_eta, stop=False, speed=speed)
        # Set waypoint list data member.
        self._core_waypoint_list = [start_wpt, end_wpt]

    def set_core_waypoints_from_UFACET_scan_pass(self, ufacet_scan_pass):
        """
        Set the core waypoints from a UFACET scan pass.

        Parameters
        ----------
        ufacet_scan_pass : UfacetScanPass
            The UFACET scan pass from which to set the core waypoints.

        Returns
        -------
        None
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # Save locale.
        self._locale = ufacet_scan_pass.ufacet_scan_parameters["locale"]
        # Set UFACET-specific data member.
        self._ufacet_scan_pass = ufacet_scan_pass
        # Set waypoint list data member.
        self._core_waypoint_list = ufacet_scan_pass.waypoints()

    def set_waypoints_with_margin(self, scan_parameters):
        """
        Set waypoints with lead-in and run-past margins.

        Parameters
        ----------
        scan_parameters : dict
            A dictionary containing parameters for the scan, including lead-in and run-past distances.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the core waypoint list is not set or has fewer than two elements.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # Check input.
        if self._core_waypoint_list == None:
            print("In ScanPass.set_waypoints_with_margin(), attempt to fetch unset _core_waypoint_list.")
            assert False
        if len(self._core_waypoint_list) < 2:
            print("In ScanPass.set_waypoints_with_margin(), _core_waypoint_list had fewer than two elements.")
            assert False
        # Fetch control parameters.
        lead_in = scan_parameters["lead_in"]
        run_past = scan_parameters["run_past"]
        # Set data members.
        self._lead_in = lead_in
        self._run_past = run_past
        # Fetch core pass endpoint coordinates.
        wpt0 = self._core_waypoint_list[0]
        xyz0 = wpt0.xyz
        x0 = xyz0[0]
        y0 = xyz0[1]
        z0 = xyz0[2]
        wptN = self._core_waypoint_list[-1]  # There might be multiple waypoints in one pass.
        xyzN = wptN.xyz
        xN = xyzN[0]
        yN = xyzN[1]
        zN = xyzN[2]
        # Compute unit vector pointing in the scan direction.
        Dx = xN - x0
        Dy = yN - y0
        Dz = zN - z0
        d = np.sqrt((Dx * Dx) + (Dy * Dy) + (Dz * Dz))
        ux = Dx / d
        uy = Dy / d
        uz = Dz / d
        # Compute start and end points, including margin.
        x_start = x0 - (lead_in * ux)
        y_start = y0 - (lead_in * uy)
        z_start = z0 - (lead_in * uz)
        x_end = xN + (run_past * ux)
        y_end = yN + (run_past * uy)
        z_end = zN + (run_past * uz)
        start_xyz = [x_start, y_start, z_start]
        end_xyz = [x_end, y_end, z_end]
        # Compute start and end gaze angles, so that gaze angles at core start and end points will be proper.
        eta0 = wpt0.eta
        etaN = wptN.eta
        Deta = etaN - eta0
        lead_in_frac = lead_in / d
        lead_in_Deta = Deta * lead_in_frac
        run_past_frac = run_past / d
        run_past_Deta = Deta * run_past_frac
        start_eta = eta0 - lead_in_Deta
        end_eta = etaN + run_past_Deta
        # Fetch locale.
        locale0 = wpt0.locale
        localeN = wptN.locale
        if locale0 != localeN:
            print(
                "In ScanPass.set_waypoints_with_margin(), mismatched locale0="
                + str(locale0)
                + " and localeN="
                + str(localeN)
                + "."
            )
            assert False
        # Fetch theta.
        theta0 = wpt0.theta
        thetaN = wptN.theta
        if theta0 != thetaN:
            print(
                "In ScanPass.set_waypoints_with_margin(), mismatched theta0="
                + str(np.rad2deg(theta0))
                + " and thetaN="
                + str(np.rad2deg(thetaN))
                + "."
            )
            assert False
        # Fetch speed.
        speed0 = wpt0.speed
        speedN = wptN.speed
        if speed0 != speedN:
            print(
                "In ScanPass.set_waypoints_with_margin(), mismatched speed0="
                + str(speed0)
                + " and speedN="
                + str(speedN)
                + "."
            )
            assert False
        # Construct start and end waypoints.
        start_wpt = wp.WayPoint(locale0, start_xyz, theta0, start_eta, stop=False, speed=speed0)
        end_wpt = wp.WayPoint(localeN, end_xyz, thetaN, end_eta, stop=False, speed=speedN)
        # Produce updated waypoint list.
        if len(self._core_waypoint_list) == 2:
            waypoint_list = [
                start_wpt,
                end_wpt,
            ]  # If there are only two waypoints, keep this property to avoid potential stutters.
        else:
            waypoint_list = [start_wpt] + self._core_waypoint_list + [end_wpt]
        # Store result.
        self._waypoint_list = waypoint_list

    # RENDERING

    def draw(self, view, scan_pass_styles):
        """
        Render the scan pass on the specified view using the provided styles.

        Parameters
        ----------
        view : object
            The view object where the scan pass will be rendered.
        scan_pass_styles : object
            An object containing styles for rendering the scan pass.

        Returns
        -------
        None
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # Fetch draw style control.
        scan_pass_style = scan_pass_styles.style(self.idx)

        # Segment of interest.
        if scan_pass_style.draw_segment_of_interest and (self._segment_dict != None):
            # Fetch segment of interest.
            segment_xyz = self._segment_dict["segment_xyz"]
            # Draw segment.
            view.draw_xyz_list(segment_xyz, style=scan_pass_style.segment_of_interest_style)

        # Core waypoint segment.
        if scan_pass_style.draw_core_segment:
            # Fetch core segment.
            core_wpt0 = self.core_waypoints()[0]
            core_wptN = self.core_waypoints()[-1]
            core_segment_xyz = [core_wpt0.xyz, core_wptN.xyz]
            # Draw segment.
            view.draw_xyz_list(core_segment_xyz, style=scan_pass_style.core_segment_style)


# -------------------------------------------------------------------------------------------------------
# CONSTRUCTION FUNCTIONS
#


def construct_scan_pass_given_segment_of_interest(segment_xyz, fly_backward, scan_parameters):
    """
    Construct a ScanPass object given a segment of interest.

    Parameters
    ----------
    segment_xyz : list of list of float
        A list of XYZ coordinates defining the segment of interest.
    fly_backward : bool
        A flag indicating whether to fly backward along the segment.
    scan_parameters : dict
        A dictionary containing parameters for the scan configuration.

    Returns
    -------
    ScanPass
        A ScanPass object constructed from the provided segment and parameters.

    Examples
    --------
    >>> scan_pass = construct_scan_pass_given_segment_of_interest([[0, 0, 0], [1, 1, 1]], False, {'locale': 'en'})
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    scan_pass = ScanPass()
    scan_pass.set_core_waypoints_given_segment_of_interest(segment_xyz, fly_backward, scan_parameters)
    scan_pass.set_waypoints_with_margin(scan_parameters)
    return scan_pass


def construct_scan_pass_given_UFACET_scan_pass(
    ufacet_scan_pass, fly_backward, scan_parameters  # UfacetScanPass object.
):
    """
    Construct a ScanPass object given a UFACET scan pass.

    Parameters
    ----------
    ufacet_scan_pass : UfacetScanPass
        The UFACET scan pass from which to construct the ScanPass.
    fly_backward : bool
        A flag indicating whether to fly backward for this scan pass.
    scan_parameters : dict
        A dictionary containing parameters for the scan configuration.

    Returns
    -------
    ScanPass
        A ScanPass object constructed from the provided UFACET scan pass and parameters.

    Examples
    --------
    >>> scan_pass = construct_scan_pass_given_UFACET_scan_pass(ufacet_pass, True, {'locale': 'en'})
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Notify progress.
    print(
        "Constructing UFACET scan pass "
        + ufacet_scan_pass.heliostat_name_list[0]
        + "-"
        + ufacet_scan_pass.heliostat_name_list[-1]
        + "..."
    )

    scan_pass = ScanPass()
    scan_pass.set_core_waypoints_from_UFACET_scan_pass(ufacet_scan_pass)
    if fly_backward:
        scan_pass.core_waypoints().reverse()
    scan_pass.set_waypoints_with_margin(scan_parameters)
    return scan_pass
