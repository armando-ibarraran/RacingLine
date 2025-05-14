
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np
from optimization_functions import rcBFGSLiMem
from extra_functions import project_and_center, densify_linestring, compute_track_border_lines, plot_racing_line, time_and_penalties

import json
import datetime


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

    car_weight : float
        Weight of the car (in kilograms), used to calculate maximum friction.

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
    N = len(cross_lines)

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
        total_time, penalty = time_and_penalties(x, cross_lines, v_max, a_min, a_max, r_min, max_lat)
        return total_time + penalty



    # ====================================================================================================
    # Optimice
    # ====================================================================================================
    # Decision Variables:
    # - N raw interpolation parameters (one per cross-line) — first N elements
    # - N-1 velocity magnitudes (one per segment between positions) — last N-1 elements
    N = len(cross_lines)
    x0 = np.ones(2 * N - 1)     # Initial guess
    x0[:N] = 0.0      # Interpolation raw values (can be 0 or random)
    x0[N:] =10.0     # Initial guess for velocity in m/s (e.g., 35 km/h)
    initial_value = f(x0)
    print("Initial guess:", initial_value)

    # Optimization
    result, _  = rcBFGSLiMem(f, x0, 100 , 5, tol=1e-5, eta=0.1, Delta_max=100.0)



    # ====================================================================================================
    # Save and plot
    # ====================================================================================================
    # Reconstruct optimized positions
    x_opt = result

    
    # Save solution
    total_time, penalty = time_and_penalties(x_opt, cross_lines, v_max, a_min, a_max, r_min, max_lat)
    data = {"point": x_opt.tolist(), "total_time": total_time, "penalty": penalty}
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"./solutions/{circuit_name.strip().lower().replace(' ', '_')}_{timestamp}_solution.json", "w") as final:
        json.dump(data, final)

    # Plot
    plot_racing_line(x_opt, circuit_name, total_time, N, cross_lines, densified_line, left_border_line, right_border_line)

    return x_opt