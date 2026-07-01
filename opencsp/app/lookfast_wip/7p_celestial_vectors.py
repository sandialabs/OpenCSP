from zoneinfo import ZoneInfo
from datetime import datetime, timezone, timedelta
import skyfield.api as skf
import numpy as np
import matplotlib.pyplot as plt


def calculate_vectors(celestial_object_name, target_location, observer_location, observation_time):
    """
    Calculates vectors from a celestial object to a target point and from the target point to an observer.

    Parameters:
        celestial_object_name (str): Name of the celestial object (e.g., "moon", "sun").
        target_location (tuple): Latitude, longitude, and elevation of the target point (degrees, meters).
        observer_location (tuple): Latitude, longitude, and elevation of the observer (degrees, meters).
        observation_time (tuple): Observation time as (year, month, day, hour, minute, second).

    Returns:
        dict: A dictionary containing the vectors:
            - "celestial_to_target": Vector from the celestial object to the target point (normalized).
            - "target_to_observer": Vector from the target point to the observer (normalized).
    """
    # Load ephemeris data (DE430 dataset)
    eph = skf.load("de430t.bsp")  # DE430 ephemeris file

    # Define celestial object
    celestial_object = eph[celestial_object_name]

    # Define target location
    target = skf.Topos(
        latitude_degrees=target_location[0], longitude_degrees=target_location[1], elevation_m=target_location[2]
    )

    # Define observer location
    observer = skf.Topos(
        latitude_degrees=observer_location[0], longitude_degrees=observer_location[1], elevation_m=observer_location[2]
    )

    # Define observation time
    ts = skf.load.timescale()
    if isinstance(observation_time, datetime):
        # Extract components from the datetime object
        observation_time = ts.utc(
            observation_time.year,
            observation_time.month,
            observation_time.day,
            observation_time.hour,
            observation_time.minute,
            observation_time.second,
        )
    else:
        # If observation_time is already a tuple, unpack it directly
        observation_time = ts.utc(*observation_time)

    # Calculate position of the celestial object relative to the target
    earth = eph["earth"]
    target_position = earth + target
    celestial_to_target = target_position.at(observation_time).observe(celestial_object).position.km

    # Normalize the vector to unit length
    celestial_to_target_normalized = celestial_to_target / np.linalg.norm(celestial_to_target)

    # Calculate position of the target relative to the observer
    observer_position = earth + observer
    target_to_observer = observer_position.at(observation_time).observe(target_position).position.km

    # Normalize the vector to unit length
    target_to_observer_normalized = target_to_observer / np.linalg.norm(target_to_observer)

    return {"celestial_to_target": celestial_to_target_normalized, "target_to_observer": target_to_observer_normalized}


def calculate_vectors_celestial_only(celestial_object_name, target_location, observation_time):
    """
    Calculates vectors from a celestial object to a target point and from the target point to an observer.

    Parameters:
        celestial_object_name (str): Name of the celestial object (e.g., "moon", "sun").
        target_location (tuple): Latitude, longitude, and elevation of the target point (degrees, meters).
        observation_time (tuple): Observation time as (year, month, day, hour, minute, second).

    Returns:
        dict: A dictionary containing the vectors:
            - "celestial_to_target": Vector from the celestial object to the target point (normalized).
            - "target_to_observer": Vector from the target point to the observer (normalized).
    """
    # Load ephemeris data (DE430 dataset)
    eph = skf.load("de430t.bsp")  # DE430 ephemeris file

    # Define celestial object
    celestial_object = eph[celestial_object_name]

    # Define target location
    target = skf.Topos(
        latitude_degrees=target_location[0], longitude_degrees=target_location[1], elevation_m=target_location[2]
    )

    # Define observation time
    ts = skf.load.timescale()
    if isinstance(observation_time, skf.Time):
        pass
    elif isinstance(observation_time, datetime):
        # Extract components from the datetime object
        observation_time = ts.utc(
            observation_time.year,
            observation_time.month,
            observation_time.day,
            observation_time.hour,
            observation_time.minute,
            observation_time.second,
        )
    else:
        # If observation_time is already a tuple, unpack it directly
        observation_time = ts.utc(*observation_time)

    # Calculate position of the celestial object relative to the target
    earth = eph["earth"]
    target_position_vec = earth + target
    target_position_bary = target_position_vec.at(observation_time)
    target_obsv_celest = target_position_bary.observe(celestial_object)
    target_obsv_celest_apparent = target_obsv_celest.apparent()

    celestial_to_target = target_position_vec.at(observation_time).observe(celestial_object).position.km
    celestial_to_target_normalized = celestial_to_target / np.linalg.norm(celestial_to_target)

    earth_target_bary = earth.at(observation_time)
    earth_obsv_target = earth_target_bary.observe(target_position_vec)
    earth_obsv_target_apparent = earth_obsv_target.apparent()

    NSTTF_coord_toca = [
        target_obsv_celest_apparent.frame_xyz(target).km[1],
        target_obsv_celest_apparent.frame_xyz(target).km[0],
        target_obsv_celest_apparent.frame_xyz(target).km[2],
    ]
    NSTTF_coord_eota = [
        earth_obsv_target_apparent.frame_xyz(target).km[1],
        earth_obsv_target_apparent.frame_xyz(target).km[0],
        earth_obsv_target_apparent.frame_xyz(target).km[2],
    ]
    # Calculate angular size in radians
    distance_to_celestial_object = target_obsv_celest_apparent.distance().km  # Distance in kilometers

    if celestial_object_name.lower() == 'moon':
        radius_of_celestial_object = 1737.4  # Radius in kilometers of the Moon
    elif celestial_object_name.lower() == 'sun':
        radius_of_celestial_object = 696340  # Radius in kilometers of the Sun
    else:
        radius_of_celestial_object = 1737.4  # Radius in kilometers of the Moon

    angular_size_radians = 2 * np.arctan(radius_of_celestial_object / distance_to_celestial_object)

    # Apparent function returns a tuple with (Altitude, Azimuth, and Distance)
    # Altitude measures the angle above or below the horizon. The zenith is at +90°, an object on the horizon’s great circle is at 0°, and the nadir beneath your feet is at −90°.
    # Azimuth measures the angle around the sky from the north pole: 0° means exactly north, 90° is east, 180° is south, and 270° is west.
    # "earth_to_target_spherical": earth_obsv_target_apparent.altaz()
    # Cartesian for horizonal skyfield coordinates is Left Handed {x points north, y points east, z points to zenith}
    # NSTTF coordinates is Right Handed {x points east, y points north, z points to zenith}
    return {
        "celestial_to_target": celestial_to_target_normalized,
        "celestial_to_target_apparent": target_obsv_celest_apparent,
        "cel_to_target_cartesian": NSTTF_coord_toca / np.linalg.norm(NSTTF_coord_toca),
        "cel_to_target_spherical": target_obsv_celest_apparent.altaz(),
        "earth_to_target_apparent": earth_obsv_target_apparent,
        "earth_to_target_cartesian": NSTTF_coord_eota / np.linalg.norm(NSTTF_coord_eota),
        "angular_size_radians": angular_size_radians,
    }


def plot_circular_projection_on_unit_sphere(vector, angular_size, label=None):
    """
    Plots the circular projection of the Sun or Moon on a unit sphere.

    Parameters:
    - vector: 3D vector (numpy array) pointing to the Sun/Moon (normalized to unit length).
    - angular_size: Angular size of the Sun/Moon in degrees.
    - label: Optional label for the projection (e.g., "Sun", "Moon").
    """
    # Convert angular size to radians and calculate angular radius
    angular_radius = np.radians(angular_size / 2)

    # Create a circle in 3D space
    num_points = 360  # Number of points to define the circle
    theta = np.linspace(0, 2 * np.pi, num_points)  # Angle around the circle

    # Find a basis for the plane perpendicular to the vector
    # First basis vector (arbitrary vector not parallel to the input vector)
    arbitrary_vector = np.array([1, 0, 0]) if not np.allclose(vector, [1, 0, 0]) else np.array([0, 1, 0])
    basis1 = np.cross(vector, arbitrary_vector)
    basis1 /= np.linalg.norm(basis1)  # Normalize

    # Second basis vector (perpendicular to both the input vector and basis1)
    basis2 = np.cross(vector, basis1)
    basis2 /= np.linalg.norm(basis2)  # Normalize

    # Generate points on the circle
    circle_points = []
    for angle in theta:
        point = np.cos(angular_radius) * vector + np.sin(angular_radius) * (
            np.cos(angle) * basis1 + np.sin(angle) * basis2
        )
        circle_points.append(point)
    circle_points = np.array(circle_points)

    # Create a unit sphere for visualization
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Initialize 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the unit sphere
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)

    # Plot the vector pointing to the Sun/Moon
    ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='green', label='Vector', arrow_length_ratio=0.1)

    # Plot the circular projection
    ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='orange', label=label)

    # Set plot limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Circular Projection on Unit Sphere')

    # Add legend
    if label:
        ax.legend()

    plt.show()


def plot_all_vectors_on_unit_sphere(vectors_dict, angular_size):
    """
    Plots all vectors and their circular projections on a unit sphere.

    Parameters:
    - vectors_dict: Dictionary containing vectors and their names as keys (e.g., {"celestial_to_target": vector1, "target_to_observer": vector2}).
    - angular_size: Angular size in degrees for the circular projection.
    """
    # Convert angular size to radians and calculate angular radius
    angular_radius = np.radians(angular_size / 2)

    # Create a unit sphere for visualization
    u = np.linspace(0, 2 * np.pi, 360)
    v = np.linspace(0, np.pi, 360)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Initialize 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the unit sphere
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.2)

    # Iterate over all vectors in the dictionary
    for label, vector in vectors_dict.items():
        # Create a circle in 3D space for the projection
        num_points = 360  # Number of points to define the circle
        theta = np.linspace(0, 2 * np.pi, num_points)  # Angle around the circle

        # Find a basis for the plane perpendicular to the vector
        arbitrary_vector = np.array([1, 0, 0]) if not np.allclose(vector, [1, 0, 0]) else np.array([0, 1, 0])
        basis1 = np.cross(vector, arbitrary_vector)
        basis1 /= np.linalg.norm(basis1)  # Normalize

        basis2 = np.cross(vector, basis1)
        basis2 /= np.linalg.norm(basis2)  # Normalize

        # Generate points on the circle
        circle_points = []
        for angle in theta:
            point = np.cos(angular_radius) * vector + np.sin(angular_radius) * (
                np.cos(angle) * basis1 + np.sin(angle) * basis2
            )
            circle_points.append(point)
        circle_points = np.array(circle_points)

        # Plot the vector
        ax.quiver(
            0, 0, 0, vector[0], vector[1], vector[2], color='green', label=f'Vector: {label}', arrow_length_ratio=0.1
        )

        # Plot the circular projection
        ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], label=f'Projection: {label}')

    # Set plot limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Vectors and Circular Projections on Unit Sphere')

    # Add legend
    ax.legend()

    plt.show()


def lat_long_to_decimal(input):
    decimal = input[0] + input[1] / 60 + input[2] / 3600
    return decimal


def interpolate_and_plot_celestial_body(
    celestial_object_name, target_location, start_time, end_time, num_interpolated_times
):
    """
    Interpolates a series of times between two provided times and plots the projection of a celestial body
    (including its angular size) on the unit sphere to visualize its traversal.

    Parameters:
        celestial_object_name (str): Name of the celestial object (e.g., "moon", "sun").
        target_location (tuple): Latitude, longitude, and elevation of the target point (degrees, meters).
        start_time (datetime): Start time as a UTC datetime object.
        end_time (datetime): End time as a UTC datetime object.
        num_interpolated_times (int): Number of interpolated times between start_time and end_time.
        angular_size (float): Angular size of the celestial body in degrees.

    Returns:
        None
    """
    # Load ephemeris data (DE430 dataset)
    eph = skf.load("de430t.bsp")  # DE430 ephemeris file

    # Define celestial object
    celestial_object = eph[celestial_object_name]

    # Define target location
    target = skf.Topos(
        latitude_degrees=target_location[0], longitude_degrees=target_location[1], elevation_m=target_location[2]
    )

    # Define observation timescale
    ts = skf.load.timescale()

    # Interpolate times between start_time and end_time
    interpolated_times = [
        start_time + i * (end_time - start_time) / (num_interpolated_times - 1) for i in range(num_interpolated_times)
    ]

    # Calculate vectors for each interpolated time
    vectors = []
    for time in interpolated_times:
        observation_time = ts.utc(time.year, time.month, time.day, time.hour, time.minute, time.second)
        result = calculate_vectors_celestial_only(
            celestial_object_name=celestial_object_name,
            target_location=target_location,
            observation_time=observation_time,
        )
        vectors.append(result)

    # Create a the top half of a unit sphere for visualization of the horizon
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi / 2, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Initialize 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the unit sphere
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)

    # Plot the projections for each interpolated time
    for time, vector in enumerate(vectors):
        # Convert angular size to radians and calculate angular radius
        angular_radius = vector['angular_size_radians'] / 2
        # Create a circle in 3D space for the projection
        num_points = 360  # Number of points to define the circle
        theta = np.linspace(0, 2 * np.pi, num_points)  # Angle around the circle

        # Find a basis for the plane perpendicular to the vector
        arbitrary_vector = (
            np.array([1, 0, 0])
            if not np.allclose(vector['cel_to_target_cartesian'], [1, 0, 0])
            else np.array([0, 1, 0])
        )
        basis1 = np.cross(vector['cel_to_target_cartesian'], arbitrary_vector)
        basis1 /= np.linalg.norm(basis1)  # Normalize

        basis2 = np.cross(vector['cel_to_target_cartesian'], basis1)
        basis2 /= np.linalg.norm(basis2)  # Normalize

        # Generate points on the circle
        circle_points = []
        for angle in theta:
            point = np.cos(angular_radius) * vector['cel_to_target_cartesian'] + np.sin(angular_radius) * (
                np.cos(angle) * basis1 + np.sin(angle) * basis2
            )
            circle_points.append(point)
        circle_points = np.array(circle_points)

        # Plot the circular projection
        ax.plot(
            circle_points[:, 0],
            circle_points[:, 1],
            circle_points[:, 2],
            label=f"Time: {interpolated_times[time]} (UTC)",
        )
    '''
    # Plot the vector
        ax.quiver(
            0,
            0,
            0,
            vector['cel_to_target_cartesian'][0],
            vector['cel_to_target_cartesian'][1],
            vector['cel_to_target_cartesian'][2],
            color='green',
            label=f'Vector: Heliostat to Celestial',
            arrow_length_ratio=0.1,
        )
        ax.quiver(
            0,
            0,
            0,
            vector['earth_to_target_cartesian'][0] * -1,
            vector['earth_to_target_cartesian'][1] * -1,
            vector['earth_to_target_cartesian'][2] * -1,
            color='k',
            label=f'Vector: Heliostat to Earth Center',
            arrow_length_ratio=0.1,
        )
    '''

    # Set plot limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel('X - East is Positive')
    ax.set_ylabel('Y - North is Positive')
    ax.set_zlabel('Z - Zenith is Positive')
    ax.set_title(f'{celestial_object_name.capitalize()} Traversal on Unit Sphere')
    ax.set_aspect('equal')
    ax.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5))

    plt.show()


# Example usage
if __name__ == "__main__":
    # Define angular size of the Sun (example: ~0.5 degrees)
    # angular_size_sun = np.degrees(0.0049)

    # Define observer location Example NSTTF Tower 260 Level Balcony
    observer_lat = (34, 57, 44.56)  # (degree, minute, second) negative degree for south or west
    observer_long = (-106, 30, 34.70)  # (degree, minute, second) negative degree for south or west
    observer_elevation = 1755.648
    observer_location = (lat_long_to_decimal(observer_lat), lat_long_to_decimal(observer_long), observer_elevation)

    # Define target location Example NSTTF Heliostat 5E9
    target_lat = (34, 57, 46.08)  # (degree, minute, second) negative degree for south or west
    target_long = (-106, 30, 31.43)  # (degree, minute, second) negative degree for south or west
    target_elevation = 1706.88
    target_location = (
        lat_long_to_decimal(target_lat),
        lat_long_to_decimal(target_long),
        target_elevation,
    )  # Example target location (latitude, longitude, elevation in meters)

    # Define observation time
    # Arguments: year, month, day, hour, minute, second
    albuquerque_tz = ZoneInfo("America/Denver")
    observation_time_local = datetime(2025, 6, 14, 4, 10, 0, tzinfo=albuquerque_tz)
    observation_time_utc = observation_time_local.astimezone(timezone.utc)  # UTC time
    observation_time_utc_end = observation_time_utc + timedelta(minutes=12)

    # Calculate vectors for the Sun
    vectors = calculate_vectors("moon", target_location, observer_location, observation_time_utc)

    # Plot the circular projection of the Sun
    # plot_all_vectors_on_unit_sphere(vectors, angular_size=angular_size_sun)

    interpolate_and_plot_celestial_body(
        "moon",
        target_location=target_location,
        start_time=observation_time_utc,
        end_time=observation_time_utc_end,
        num_interpolated_times=10,
    )
