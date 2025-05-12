
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np
from functions import project_and_center, densify_linestring, compute_track_border_lines, plot_racing_line


def get_raceline(geojson_path, circuit_name, circuit_density, circuit_width, v_max, a_min, a_max, r_min, mu):
    """
    Computes the optimal racing line for a given circuit using physical and geometric constraints.

    This function loads a circuit layout from a GeoJSON file, densifies its centerline,
    constructs cross-line slices, and optimizes the trajectory by minimizing lap time
    under constraints such as velocity bounds, acceleration limits, turning radius,
    and lateral friction.

    Parameters
    ----------
    geojson_path : str
        Path to the GeoJSON file containing the circuit geometry.

    circuit_name : str
        Name of the target circuit within the GeoJSON file.

    circuit_density : float
        Distance (in meters) between trajectory points along the centerline.

    circuit_width : float
        Width of the track (in meters), used to construct track borders and valid cross-lines.

    v_max : float
        Maximum allowable velocity (in m/s) for the car.

    a_min : float
        Minimum longitudinal acceleration (in m/s²), i.e., maximum braking.

    a_max : float
        Maximum longitudinal acceleration (in m/s²), i.e., maximum propulsion.

    r_min : float
        Minimum allowable turning radius (in meters), based on vehicle geometry.

    mu : float
        Coefficient of lateral friction for grip-based cornering (typical range: 1.2–1.5 for race tires).

    Returns
    -------
    x_opt : ndarray
        Optimized decision variable vector:
        - First N values: interpolation parameters (raw)
        - Last N-1 values: velocities (in m/s) along each segment of the racing line
    """

    # ====================================================================================================
    # Get Data
    # ====================================================================================================
    # Load the .geojson file
    gdf = gpd.read_file(geojson_path)

    # Select the desired circuit
    circuit = gdf[gdf['Name'] == circuit_name]

    # Project to meters and center the geometry
    projected_circuit = project_and_center(circuit)

    # Get the original line geometry
    circuit_line = projected_circuit.geometry.iloc[0]

    # Densify
    densified_line = densify_linestring(circuit_line, circuit_density)

    # Set 12-meter wide track
    left_border_line, right_border_line = compute_track_border_lines(densified_line, circuit_width)

    # Create cross-lines
    cross_lines = [LineString([left, right]) for left, right in zip(left_border_line.coords, right_border_line.coords)]

    # Calculate max allowed lateral g-force
    max_lat = mu * 9.81  



    # ====================================================================================================
    # Define decision variables and objective function
    # ====================================================================================================
    # Decision Variables:
    # - N raw interpolation parameters (one per cross-line) — first N elements
    # - N-1 velocity magnitudes (one per segment between positions) — last N-1 elements
    N = len(cross_lines)
    x0 = np.ones(2 * N - 1)*0.5  # [interpolations (raw, R), velocities (>0)]

    # Objective function to minimize
    def f(x):
        """
        Computes the total lap time for a car racing line, with soft penalties for physical constraints.
        
        Decision Variables:
        - First N values in `x`: interpolation parameters (real-valued, mapped to (0, 1) via sigmoid)
        - Last N-1 values in `x`: velocities at each segment (in m/s)

        Returns:
            float: Total lap time (seconds) + penalty term for constraint violations.
        """
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

        for i in range(N - 1):
            dist = positions[i].distance(positions[i + 1])
            v1 = velocity[i]

            # --- 1. Velocity constraints ---
            penalty += np.exp(2 * (-v1)) * 5 
            penalty += np.exp(2 * (v1 - v_max)) * 5  

            if i < N - 2:
                # --- 2. Acceleration penalty (based on segment change) ---
                v2 = velocity[i + 1]
                next_dist = positions[i + 1].distance(positions[i + 2])
                avg_dist = 0.5 * (dist + next_dist)
                accel = (v2**2 - v1**2) / (2 * avg_dist)
                penalty += np.exp(10 * (accel - a_max)) * 10
                penalty += np.exp(10 * (a_min - accel)) * 10

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
                    area = np.sqrt(area_sq)
                    radius = (a * b * c) / (4 * area)
                penalty += np.exp(2 * (r_min - radius)) * 20

                # --- 4. Lateral acceleration limit (friction constraint) ---
                lateral_accel = v1**2 / radius
                penalty += np.exp(10 * (lateral_accel - max_lat)) * 10

            # --- 5. Time contribution ---
            total_time += dist / v1

        return total_time + penalty



    # ====================================================================================================
    # Optimice
    # ====================================================================================================
    from scipy.optimize import minimize

    N = len(cross_lines)

    # Initial guess
    x0 = np.ones(2 * N - 1)
    x0[:N] = 0.0      # Interpolation raw values (can be 0 or random)
    x0[N:] = 10.0     # Initial guess for velocity in m/s (e.g., 35 km/h)
    initial_value = f(x0)
    print("Initial guess:", initial_value)

    # Optimization
    result = minimize(
        fun=f,
        x0=x0,
        method='L-BFGS-B',
        options={'maxiter': 500, 'disp': True}
    )



    # ====================================================================================================
    # Plot
    # ====================================================================================================
    # Reconstruct optimized positions
    x_opt = result.x
    plot_racing_line(x_opt, circuit_name, N, cross_lines, densified_line, left_border_line, right_border_line)


    return x_opt