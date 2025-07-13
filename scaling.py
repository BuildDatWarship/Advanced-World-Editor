# advanced-world-editor/scaling.py

import numpy as np
from constants import METERS_TO_FEET, KELVIN_TO_CELSIUS_OFFSET

class ScalingManager:
    """Handles conversions between normalized (0-1) space and real-world units."""

    def __init__(self, app):
        self.app = app

    def to_real(self, normalized_value):
        """Converts a 0-1 value to the current real-world unit (m or ft)."""
        max_height_m = self.app.vars["max_world_height_m"].get()
        real_val_m = normalized_value * max_height_m
        if self.app.vars["unit_system"].get() == "Imperial":
            return real_val_m * METERS_TO_FEET
        return real_val_m

    def to_normalized(self, real_value):
        """Converts a real-world value (in current units) back to the 0-1 scale."""
        max_height_m = self.app.vars["max_world_height_m"].get()
        if max_height_m == 0:
            return 0.0

        real_val_m = real_value
        if self.app.vars["unit_system"].get() == "Imperial":
            real_val_m = real_value / METERS_TO_FEET

        return np.clip(real_val_m / max_height_m, 0.0, 1.0)

    def get_unit_suffix(self):
        """Returns the appropriate unit suffix."""
        return "m" if self.app.vars["unit_system"].get() == "Metric" else "ft"

    def get_max_real_height(self):
        """Gets the max world height in the current unit system."""
        max_m = self.app.vars["max_world_height_m"].get()
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