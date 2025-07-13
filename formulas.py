# advanced-world-editor/formulas.py

# Default mathematical formulas used for world generation.
# Users can override these in the "Formulas" panel of the UI.

# The symbol table for these formulas will include:
# - cm: continent_base map
# - tsm: tectonic_smoothed map
# - rm: mountain_ridges map
# - bn: boundary_noise map
# - hmap: final heightmap
# - albedo: albedo_map
# - lat_rad: latitude in radians (for temp map)
# - height_m: height in meters (for temp map)
# - sea_level_m: sea level in meters (for temp map)
# - flow_angles: wind/flow direction in radians (for rain map)
# - dx, dy: heightmap gradient components (for rain map)
#
# All parameters from the UI are also available (e.g., uplift_magnitude, lapse_rate_c_per_1000m, etc.)

DEFAULT_FORMULAS = {
    "heightmap_formula": "(cm + (tsm * uplift_magnitude) + (rm * tsm * ridge_strength) + (bn * tsm * boundary_jaggedness))",

    "temperature_base_k_formula": "((1 - albedo_mean) * solar_intensity * (1/pi) / stefan_boltzmann)**0.25",
    
    "temperature_greenhouse_k_formula": "base_k + (5.35 * log(1 + co2_level_ppm / 280.0)) + (20.0 * log(1.0 + global_rainfall / 985.5))",

    "temperature_latitude_gradient_formula": "pole_k + (equator_k - pole_k) * cos(lat_rad)",
    
    "temperature_lapse_rate_term_formula": "to_real(np.maximum(hmap - sea_level, 0)) / 1000.0 * lapse_rate_c_per_1000m",

    "rainfall_orographic_effect_formula": "-( (cos(flow_angles) * dx) + (sin(flow_angles) * dy) ) * orographic_strength * 1000",
}