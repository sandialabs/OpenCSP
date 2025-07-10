""" """

import numpy as np

import opencsp.common.lib.geo.lon_lat_nsttf as lln


class WayPoint:
    """
    A waypoint in a flight plan.

    This class represents a waypoint defined by its geographic location, heading, gaze angle,
    and other parameters relevant to a flight operation.

    Attributes
    ----------
    idx : int
        The index of the waypoint in the flight plan.
    lon : float
        The longitude of the waypoint, calculated based on the locale.
    lat : float
        The latitude of the waypoint, calculated based on the locale.
    """

    # "ChatGPT 4o" assisted with generating this docstring.

    def __init__(
        self,
        locale,  # Information needed to convert (x,y,z into global (longitude, latitude) coordinates.
        xyz,  # m.     (x,y,z) coordinates in solar field coordinate system.
        theta,  # rad.   Heading, measured ccw from x axis (East).
        eta,  # rad.   Gaze angle, measured ccw from drone axis pointing forward.
        stop,  # bool.  Whether to stop at this waypoint.
        speed,
    ):  # m/sec. Speed to next waypoint, in heading direction.
        """
        A waypoint in a flight plan.

        This class represents a waypoint defined by its geographic location, heading, gaze angle,
        and other parameters relevant to a flight operation.

        Parameters
        ----------
        locale : str
            Information needed to convert (x, y, z) coordinates into global (longitude, latitude) coordinates.
        xyz : list of float
            The (x, y, z) coordinates in the solar field coordinate system, measured in meters.
        theta : float
            The heading of the waypoint, measured in radians counterclockwise from the x-axis (East).
        eta : float
            The gaze angle of the waypoint, measured in radians counterclockwise from the drone axis pointing forward.
        stop : bool
            A flag indicating whether the drone should stop at this waypoint.
        speed : float
            The speed to the next waypoint, in meters per second, in the heading direction.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        super(WayPoint, self).__init__()

        # Input parameters.
        self.idx = -1  # Integer.  Not set yet.
        self.locale = locale
        self.xyz = xyz
        self.theta = theta
        self.eta = eta
        self.stop = stop
        self.speed = speed

        # Dependent parameters.
        self.set_longitude_latitude()

    def set_longitude_latitude(self):
        """
        Set the longitude and latitude of the waypoint based on its coordinates and locale.

        This method uses the `locale` attribute to determine how to convert the (x, y) coordinates
        into longitude and latitude. It raises an assertion error if an unexpected locale is encountered.

        Returns
        -------
        None
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        if self.locale == 'NSTTF':
            lon, lat = lln.nsttf_lon_lat_given_xy(self.xyz[0], self.xyz[1])
            self.lon = lon
            self.lat = lat
        else:
            print("In WayPoint. set_longitude_latitude(), unexpected locale encountered.")
            assert False

    def heading_deg(self):
        """
        Get the heading of the waypoint in degrees clockwise from north.

        Returns
        -------
        float
            The heading of the waypoint in degrees.

        Examples
        --------
        >>> waypoint = WayPoint('NSTTF', [0, 0, 0], np.pi/4, 0, False, 5)
        >>> waypoint.heading_deg()
        45.0
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # For flight plan output.
        # Heading is given in degrees clockwise from north.
        waypoint_heading_deg = 450.0 - np.degrees(self.theta)
        if waypoint_heading_deg >= 360.0:
            waypoint_heading_deg -= 360.0
        return waypoint_heading_deg

    def gimbal_pitch_deg(self):
        """
        Get the gimbal pitch of the waypoint in degrees.

        The gimbal pitch is defined as 0 degrees for horizontal and -90 degrees for straight down.

        Returns
        -------
        float
            The gimbal pitch of the waypoint in degrees.

        Examples
        --------
        >>> waypoint = WayPoint('NSTTF', [0, 0, 0], 0, -np.pi/2, False, 5)
        >>> waypoint.gimbal_pitch_deg()
        -90.0
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # For flight plan output.
        # Gimbal pitch is in degrees: 0 = horizontal, -90 = straight down.
        return np.rad2deg(self.eta)

    # RENDERING

    def heading_ray(self, scale):
        """
        Construct a vector representing the heading direction of the waypoint.

        Parameters
        ----------
        scale : float
            A scaling factor to determine the length of the heading ray.

        Returns
        -------
        list
            A list containing the tail and head coordinates of the heading ray.

        Examples
        --------
        >>> waypoint = WayPoint('NSTTF', [0, 0, 0], 0, 0, False, 5)
        >>> waypoint.heading_ray(1.0)
        [[0, 0, 0], [1.0, 0.0, 0.0]]
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # Constructs the head and tail of a vector pointing in the heading direction,
        # placed at the waypoint position, and with length proportional to speed.
        length = scale * self.speed
        tail = self.xyz
        dx = length * np.cos(self.theta)
        dy = length * np.sin(self.theta)
        head = tail + np.array([dx, dy, 0])
        ray = [tail, head]
        return ray

    def gaze_ray(self, length):
        """
        Construct a vector representing the gaze direction of the waypoint.

        Parameters
        ----------
        length : float
            The length of the gaze ray.

        Returns
        -------
        list
            A list containing the tail and head coordinates of the gaze ray.

        Examples
        --------
        >>> waypoint = WayPoint('NSTTF', [0, 0, 0], 0, np.pi/4, False, 5)
        >>> waypoint.gaze_ray(1.0)
        [[0, 0, 0], [0.7071067811865476, 0.7071067811865475, 0.7071067811865475]]
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # Constructs the head and tail of a vector of given length pointing in the gaze direction,
        # placed at the waypoint position.
        tail = self.xyz
        dz = length * np.sin(self.eta)
        r = length * np.cos(self.eta)
        dx = r * np.cos(self.theta)
        dy = r * np.sin(self.theta)
        head = tail + np.array([dx, dy, dz])
        ray = [tail, head]
        return ray

    def draw(self, view, waypoint_styles):
        """
        Render the waypoint on the specified view using the provided styles.

        Parameters
        ----------
        view : object
            The view object where the waypoint will be rendered.
        waypoint_styles : object
            An object containing styles for rendering the waypoint.

        Returns
        -------
        None
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # Fetch draw style control.
        waypoint_style = waypoint_styles.style(self.idx)

        # Position.
        if waypoint_style.draw_position:
            view.draw_xyz(self.xyz, style=waypoint_style.position_style)

        # Stop.
        if waypoint_style.draw_stop and self.stop:
            view.draw_xyz(self.xyz, style=waypoint_style.stop_style)

        # idx.
        if waypoint_style.draw_idx:
            view.draw_xyz_text(self.xyz, str(self.idx), style=waypoint_style.idx_style)

        # Heading.
        if waypoint_style.draw_heading:
            # Construct ray.
            heading_ray = self.heading_ray(waypoint_style.heading_scale)
            # Draw ray.
            view.draw_xyz_list(heading_ray, style=waypoint_style.heading_style)

        # Gaze.
        if waypoint_style.draw_gaze:
            # Construct ray.
            gaze_ray = self.gaze_ray(waypoint_style.gaze_length)
            # Draw ray.
            view.draw_xyz_list(gaze_ray, style=waypoint_style.gaze_style)
