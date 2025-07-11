# advanced-world-editor/constants.py

METERS_TO_FEET = 3.28084
STEFAN_BOLTZMANN = 5.670374419e-8
KELVIN_TO_CELSIUS_OFFSET = 273.15
SEA_LEVEL = 0.4  # Default normalized sea level

# --- FIX: Overhauled biome definitions for better coverage and more realistic distribution ---
# The order is important: more specific biomes are checked first.
DEFAULT_BIOMES = [
    # Water Biomes (Special Case)
    {"name": "OCEAN", "color": "#05175c", "is_water": True},
    {"name": "INLAND_WATER", "color": "#1663b8", "is_water": True},

    # Land Biomes (ordered from specific/extreme to general)
    {"name": "ICE_CAP", "color": "#ffffff", "temp_min": -100, "temp_max": -10, "rain_min": 0, "rain_max": 1000},
    {"name": "SCORCHED", "color": "#5e2908", "temp_min": 50, "temp_max": 100, "rain_min": 0, "rain_max": 100},
    {"name": "TUNDRA", "color": "#6b705c", "temp_min": -10, "temp_max": 4, "rain_min": 100, "rain_max": 400},
    
    {"name": "TEMPERATE_RAINFOREST", "color": "#16520a", "temp_min": 4, "temp_max": 20, "rain_min": 2000, "rain_max": 10000},
    {"name": "TROPICAL_RAINFOREST", "color": "#0d4f0c", "temp_min": 20, "temp_max": 50, "rain_min": 2500, "rain_max": 10000},

    {"name": "TROPICAL_FOREST", "color": "#2a7d29", "temp_min": 20, "temp_max": 50, "rain_min": 1500, "rain_max": 2500},
    {"name": "TEMPERATE_FOREST", "color": "#459416", "temp_min": 5, "temp_max": 20, "rain_min": 800, "rain_max": 2000},
    {"name": "TAIGA", "color": "#0a4034", "temp_min": -5, "temp_max": 5, "rain_min": 400, "rain_max": 1000},

    {"name": "GRASSLAND", "color": "#9eb328", "temp_min": 0, "temp_max": 25, "rain_min": 250, "rain_max": 800},
    {"name": "SAVANNA", "color": "#b3a228", "temp_min": 18, "temp_max": 50, "rain_min": 400, "rain_max": 1500},
    
    {"name": "HOT_DESERT", "color": "#ede299", "temp_min": 25, "temp_max": 50, "rain_min": 0, "rain_max": 400},
    {"name": "COLD_DESERT", "color": "#a8a8a8", "temp_min": -50, "temp_max": 5, "rain_min": 0, "rain_max": 250},
    
    # Fallback/General Biome. Should only appear in unclassified temperature/rainfall ranges.
    {"name": "ROCK", "color": "#666666", "temp_min": -100, "temp_max": 100, "rain_min": 0, "rain_max": 10000},
]