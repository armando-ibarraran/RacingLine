
from math import floor
from shapely.affinity import translate
import numpy as np
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import datetime
from matplotlib.collections import LineCollection



def project_and_center(gdf):
    """
    Projects a GeoDataFrame to its appropriate UTM zone and centers the geometry at (0,0).
    
    Parameters:
        gdf (GeoDataFrame): GeoDataFrame in EPSG:4326 (WGS84)
    
    Returns:
        GeoDataFrame: Projected and centered
    """
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError("Input GeoDataFrame must be in EPSG:4326 (WGS84)")

    # Use first geometry to determine UTM zone
    lon, lat = gdf.geometry.iloc[0].xy[0][0], gdf.geometry.iloc[0].xy[1][0]
    zone_number = floor((lon + 180) / 6) + 1
    hemisphere = 326 if lat >= 0 else 327
    epsg_code = hemisphere * 100 + zone_number

    # Project to meters
    gdf_proj = gdf.to_crs(epsg=epsg_code)

    # Compute centroid now that CRS is projected
    center = gdf_proj.geometry.centroid.iloc[0]

    # Translate geometry to center it at (0, 0)
    gdf_proj['geometry'] = gdf_proj.geometry.apply(lambda geom: translate(geom, xoff=-center.x, yoff=-center.y))

    return gdf_proj



def densify_linestring(line, resolution=1.0):
    """
    Adds points to a LineString at regular intervals (in projected units, e.g. meters).
    
    Parameters:
        line (LineString): A Shapely LineString (projected, in meters).
        resolution (float): Distance between points (in meters).
        
    Returns:
        LineString: A new densified LineString.
    """
    length = line.length
    num_points = int(length // resolution) + 1
    distances = np.linspace(0, length, num=num_points)
    points = [line.interpolate(distance) for distance in distances]
    return LineString(points)



def compute_track_border_lines(centerline, circuit_width):
    """
    Computes left and right border LineStrings based on a centerline.
    
    Parameters:
        centerline (LineString): Densified centerline.
        circuit_width (float): Width of the track in meters.
        
    Returns:
        GeoDataFrame: With two rows, 'side' column ('left' or 'right') and LineString geometry.
    """
    coords = list(centerline.coords)
    num_coords = len(coords)
    half_width = circuit_width / 2
    left_border = []
    right_border = []

    for i in range(len(coords)):
        p_prev = np.array(coords[i - 1])
        p_next = np.array(coords[(i + 1)%num_coords])
        p_curr = np.array(coords[i])

        # Compute tangent vector
        tangent = p_next - p_prev
        tangent_length = np.linalg.norm(tangent)

        if tangent_length == 0:
            continue  # skip degenerate cases

        # Unit perpendicular vector
        perp = np.array([-tangent[1], tangent[0]]) / tangent_length

        # Compute offset points
        left_point = p_curr + perp * half_width
        right_point = p_curr - perp * half_width

        left_border.append(tuple(left_point))
        right_border.append(tuple(right_point))

    # Create LineStrings
    left_line = LineString(left_border)
    right_line = LineString(right_border)

    return left_line, right_line

def sigmoid(z):
    """
    Applies the sigmoid activation function to the input.

    The sigmoid function maps any real-valued input to the range (0, 1),
    and is defined as: σ(z) = 1 / (1 + exp(-z))

    Parameters
    ----------
    z : float or ndarray
        Input value or array of values.

    Returns
    -------
    float or ndarray
        Sigmoid-transformed output, same shape as input.
    """
    return 1 / (1 + np.exp(-z))


def plot_racing_line(x, circuit_name, total_time, N, cross_lines, densified_line, left_border_line, right_border_line):
    """
    Plots the optimized racing line colored by velocity over the track layout.

    The function reconstructs the trajectory based on decision variables from the optimizer,
    interpolates positions across the track's cross-lines, and visualizes the trajectory
    using a color map where color encodes velocity.

    Parameters
    ----------
    x : ndarray
        Decision variable vector containing:
        - First N values: interpolation parameters (raw, mapped to (0,1) via sigmoid)
        - Last N-1 values: velocity values (in m/s) at each segment.

    circuit_name : str
        Name of the circuit, used for plot title and file name.

    total_time : float
        Time that takes the car to complete the lap.

    N : int
        Number of cross-lines (interpolated trajectory points).

    cross_lines : list of LineString
        List of Shapely cross-lines across the track, used to compute interpolated positions.

    densified_line : LineString
        The centerline of the circuit, used as a reference path (usually gray).

    left_border_line : LineString
        LineString representing the left border of the track.

    right_border_line : LineString
        LineString representing the right border of the track.

    Saves
    -----
    SVG plot file named with timestamp, showing the optimal racing line colored by velocity.

    Displays
    --------
    A matplotlib plot of the track and optimal line.
    """

    # Split and transform decision variables
    interpolation_raw = x[:N]
    velocity = x[N:]
    interpolation = 1 / (1 + np.exp(-interpolation_raw))

    # Interpolated optimal positions
    optimal_positions = [
        cross_lines[i].interpolate(interpolation[i], normalized=True)
        for i in range(N)
    ]
    coords = np.array([[pt.x, pt.y] for pt in optimal_positions])

    # Create segments for LineCollection
    segments = np.array([[coords[i], coords[i + 1]] for i in range(N - 1)])

    # Set up plot
    fig, ax = plt.subplots(figsize=(20, 20))

    # Plot borders and centerline
    ax.plot(*densified_line.xy, color='gray', linewidth=1, label='Centerline')
    ax.plot(*left_border_line.xy, color='black', linewidth=1)
    ax.plot(*right_border_line.xy, color='black', linewidth=1)

    # Convert velocity from m/s to km/h
    velocity_kmh = velocity * 3.6

    # Plot racing line with velocity-based color in km/h
    lc = LineCollection(segments, array=velocity_kmh, cmap='viridis', linewidth=2, label='Racing Line')
    line = ax.add_collection(lc)
    cbar = plt.colorbar(line, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Velocity (km/h)")

    # Mark start and end of the racing line
    ax.scatter(*coords[0], color='red', s=100, marker='x', label='Start')

    # Final plot formatting
    ax.set_title(f"{circuit_name}\nTotal Lap Time: {total_time:.2f} s", fontsize=16)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True)
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"./racing_lines/{circuit_name.strip().lower().replace(' ', '_')}_{timestamp}_racing_line.svg")
    plt.show()

 # Objective function to minimize
def time_and_penalties(x, cross_lines, v_max, a_min, a_max, r_min, max_lat):
    """
    Computes the total lap time for a car racing line and the penalties for physical constraints.
    
    Decision Variables:
    - First N values in `x`: interpolation parameters (real-valued, mapped to (0, 1) via sigmoid)
    - Last N-1 values in `x`: velocities at each segment (in m/s)

    Returns:
        tuple: Total lap time (seconds) and penalty term for constraint violations.
    """
    N = len(cross_lines)
    assert len(x) == 2 * N - 1, "x must have N interpolation parameters and N-1 velocities"

    # Split decision variables
    interpolation_raw = x[:N]
    velocity = x[N:]

    # Sigmoid mapping to (0, 1)
    interpolation = 1 / (1 + np.exp(-interpolation_raw))

    # Interpolate positions across each cross-line
    positions = []
    for i in range(N):
        t = interpolation[i]
        pos = cross_lines[i].interpolate(t, normalized=True)
        positions.append(pos)

    total_time = 0.0
    penalty = 0.0  # Accumulate soft penalties here

    # Penalize and compute time for each point
    for i in range(N - 1):
        dist = positions[i].distance(positions[i + 1])
        v1 = velocity[i]

        # --- 1. Velocity constraints ---
        if v1<=0:
            penalty += np.exp(2 * (-v1)) * 2 - 2
        if v1>v_max:
            penalty += np.exp(2 * (v1 - v_max)) * 2 - 2

        if i < N - 2:
            # --- 2. Acceleration penalty (based on segment change) ---
            v2 = velocity[i + 1]
            next_dist = positions[i + 1].distance(positions[i + 2])
            avg_dist = 0.5 * (dist + next_dist)
            accel = (v2**2 - v1**2) / (2 * avg_dist)
            if accel > a_max:
                penalty += np.exp(2 * (accel - a_max)) * 2 - 2
            if accel < a_min:
                penalty += np.exp(2 * (a_min - accel)) * 2 - 2
            # Penalize start point (can't start with high speeds)
            if i==0:
                accel = (velocity[0]**2 ) / (2 * dist)
                if accel > a_max:
                    penalty += np.exp(2 * (accel - a_max)) * 2 - 2
                if accel < a_min:
                    penalty += np.exp(2 * (a_min - accel)) * 2 - 2

            # --- 3. Turning radius penalty (geometric limit) ---
            p1, p2, p3 = positions[i], positions[i + 1], positions[i + 2]
            a = p1.distance(p2)
            b = p2.distance(p3)
            c = p3.distance(p1)
            s = 0.5 * (a + b + c)
            area_sq = s * (s - a) * (s - b) * (s - c)
            if area_sq <= 0: 
                radius = np.inf     # Handle collinear points
            else:
                radius = (a * b * c) / (4 * np.sqrt(area_sq))
            if radius < r_min:
                penalty += np.exp(2 * (r_min - radius)) * 2 - 2

            # --- 4. Lateral acceleration limit (friction constraint) ---
            lateral_accel = v1**2 / radius
            if lateral_accel > max_lat:
                penalty += np.exp(2 * (lateral_accel - max_lat)) * 2 - 2

        # --- 5. Time contribution ---
        total_time += dist / v1

    return total_time, penalty



