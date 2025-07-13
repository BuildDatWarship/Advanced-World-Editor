# advanced-world-editor/scaling.py

import numpy as np
from constants import METERS_TO_FEET, KELVIN_TO_CELSIUS_OFFSET

class ScalingManager:
    """Handles conversions between normalized (0-1) space and real-world units."""

    def __init__(self, app):
        self.app = app

    def to_real(self, normalized_value, above_sea=True):
        """Converts a normalized distance from sea level to meters.

        Parameters
        ----------
        normalized_value : array-like
            Distance from sea level expressed in the 0-1 normalized space.
        above_sea : bool, optional
            If ``True`` the value is treated as height above sea level,
            otherwise it is interpreted as depth below sea level.
        """
        sea_level = self.app.vars["sea_level"].get()
        elev_range = self.app.vars["elevation_range_m"].get()
        land_max = elev_range * (8848.0 / 19848.0)
        ocean_depth = elev_range - land_max

        arr = np.asarray(normalized_value, dtype=np.float32)
        if above_sea:
            real_val_m = (arr / max(1e-9, 1 - sea_level)) * land_max
        else:
            real_val_m = (arr / max(1e-9, sea_level)) * ocean_depth
        if self.app.vars["unit_system"].get() == "Imperial":
            return real_val_m * METERS_TO_FEET
        return real_val_m

    def to_normalized(self, real_value, above_sea=True):
        """Converts a real-world distance (in current units) to normalized form."""

        elev_range = self.app.vars["elevation_range_m"].get()
        if elev_range == 0:
            return 0.0

        real_val_m = real_value
        if self.app.vars["unit_system"].get() == "Imperial":
            real_val_m = real_value / METERS_TO_FEET

        sea_level = self.app.vars["sea_level"].get()
        land_max = elev_range * (8848.0 / 19848.0)
        ocean_depth = elev_range - land_max
        
        scale = land_max if above_sea else ocean_depth
        frac = np.asarray(real_val_m, dtype=np.float32) / max(1e-9, scale)
        return np.clip(frac * ((1 - sea_level) if above_sea else sea_level), 0.0, 1.0)
    
    def get_unit_suffix(self):
        """Returns the appropriate unit suffix."""
        return "m" if self.app.vars["unit_system"].get() == "Metric" else "ft"

    def get_max_real_height(self):
        """Gets the max world height in the current unit system."""
        max_m = self.app.vars["elevation_range_m"].get()
        return (
            max_m
            if self.app.vars["unit_system"].get() == "Metric"
            else max_m * METERS_TO_FEET
        )
    
    def to_display_temp(self, temp_kelvin):
        """Converts a Kelvin value to the current display unit (C or F)."""
        temp_c = temp_kelvin - KELVIN_TO_CELSIUS_OFFSET
        if self.app.vars["unit_system"].get() == "Imperial":
            temp_f = temp_c * (9 / 5) + 32
            return temp_f, "°F"
        return temp_c, "°C"

    def to_kelvin_from_display(self, temp_display):
        """Converts a display temperature (C or F) to Kelvin."""
        if self.app.vars["unit_system"].get() == "Imperial": # Input is F
            temp_c = (temp_display - 32) * (5 / 9)
        else: # Input is C
            temp_c = temp_display
        return temp_c + KELVIN_TO_CELSIUS_OFFSET