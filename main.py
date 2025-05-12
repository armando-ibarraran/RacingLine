from raceline_finder import get_raceline



# ====================================================================================================
# Track 1
# ====================================================================================================
geojson_path = "./f1-circuits.geojson"
circuit_name = 'Autódromo Hermanos Rodríguez'
circuit_density = 30.0  # in meters
circuit_width = 12.0  # in meters
v_max = 90.0  # in m/s
a_min = -10.0  # in m/s^2
a_max = 10.0  # in m/s^2
r_min = 10.0  # in meters
mu = 1.2


x_opt = get_raceline(geojson_path, circuit_name, circuit_density, circuit_width, v_max, a_min, a_max, r_min, mu)
