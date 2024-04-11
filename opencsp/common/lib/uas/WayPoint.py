"""


"""

import numpy as np

import opencsp.common.lib.geo.lon_lat_nsttf as lln


class WayPoint:
    """
    WayPoint in a flight plan.
    """

    def __init__(
        self,
        locale,  # Information needed to convert (x,y,z into global (longitude, latitude) coordinates.
        xyz,  # m.     (x,y,z) coordinates in solar field coordinate system.
        theta,  # rad.   Heading, measured ccw from x axis (East).
        eta,  # rad.   Gaze angle, measured ccw from drone axis pointing forward.
        stop,  # bool.  Whether to stop at this waypoint.
        speed,
    ):  # m/sec. Speed to next waypoint, in heading direction.
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
        if self.locale == 'NSTTF':
            lon, lat = lln.nsttf_lon_lat_given_xy(self.xyz[0], self.xyz[1])
            self.lon = lon
            self.lat = lat
        else:
            print('In WayPoint. set_longitude_latitude(), unexpected locale encountered.')
            assert False

    def heading_deg(self):
        # For flight plan output.
        # Heading is given in degrees clockwise from north.
        waypoint_heading_deg = 450.0 - np.degrees(self.theta)
        if waypoint_heading_deg >= 360.0:
            waypoint_heading_deg -= 360.0
        return waypoint_heading_deg

    def gimbal_pitch_deg(self):
        # For flight plan output.
        # Gimbal pitch is in degrees: 0 = horizontal, -90 = straight down.
        return np.rad2deg(self.eta)

    # RENDERING

    def heading_ray(self, scale):
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
