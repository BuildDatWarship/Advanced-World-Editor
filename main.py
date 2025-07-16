# advanced-world-editor/main.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, label
import threading
import queue
from ttkthemes import ThemedTk
import random
import os
import json
from asteval import Interpreter as AstevalInterpreter

# --- Local Project Imports ---
from constants import DEFAULT_BIOMES, SEA_LEVEL
from formulas import DEFAULT_FORMULAS
from scaling import ScalingManager
from history import HistoryManager, PaintAction
from ui_components import ZoomableCanvas, Tooltip, BiomeDialog
import generation as gen

class MapGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced World Editor")
        self.root.geometry("1800x1000")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.style = ttk.Style()
        self.root.style.configure("Map.TFrame", background="#282828")

        self.vars = {
            "seed": tk.IntVar(value=random.randint(0, 10000)),
            "size": tk.IntVar(value=512),
            "continent_scale": tk.DoubleVar(value=200.0),
            "flow_distortion": tk.DoubleVar(value=70.0),
            "continent_octaves": tk.IntVar(value=6),
            "plate_points": tk.IntVar(value=7),
            "uplift_magnitude": tk.DoubleVar(value=0.6),
            "tectonic_smoothing": tk.DoubleVar(value=3.0),
            "hypsometric_strength": tk.DoubleVar(value=1.0),
            "ridge_strength": tk.DoubleVar(value=0.4),
            "ridge_scale": tk.DoubleVar(value=60.0),
            "ridge_octaves": tk.IntVar(value=6),
            "boundary_jaggedness": tk.DoubleVar(value=0.05),
            "boundary_detail_scale": tk.DoubleVar(value=25.0),
            "particle_count": tk.IntVar(value=8000),
            "particle_steps": tk.IntVar(value=500),
            "particle_fade": tk.DoubleVar(value=0.99),
            "flow_scale": tk.DoubleVar(value=0.015),
            "flow_octaves": tk.IntVar(value=4),
            "unit_system": tk.StringVar(value="Metric"),
            "elevation_range_m": tk.IntVar(value=19848),
            "river_strength_threshold": tk.DoubleVar(value=0.4),
            "enable_erosion": tk.BooleanVar(value=True),
            "erosion_height_threshold": tk.DoubleVar(value=0.45),
            "erosion_sharpness": tk.DoubleVar(value=5.0),
            "global_rainfall": tk.DoubleVar(value=985.5),
            "erodibility_scale": tk.DoubleVar(value=40.0),
            "erodibility_octaves": tk.IntVar(value=4),
            "max_erosion_depth_m": tk.DoubleVar(value=250.0),
            "solar_intensity": tk.DoubleVar(value=1361.0),
            "lapse_rate_c_per_1000m": tk.DoubleVar(value=6.5),
            "latitude_north_pole": tk.DoubleVar(value=90.0),
            "latitude_south_pole": tk.DoubleVar(value=-90.0),
            "sea_level": tk.DoubleVar(value=SEA_LEVEL),
            "base_latitude_rain": tk.DoubleVar(value=1500.0),
            "orographic_strength": tk.DoubleVar(value=1.5),
            "rain_shadow_strength_factor": tk.DoubleVar(value=1.0), # Multiplier for the strength of the rain shadow effect.
            "co2_level_ppm": tk.DoubleVar(value=420.0),
            "climate_smoothing_sigma": tk.DoubleVar(value=10.0),
            "equator_pole_temp_diff_c": tk.DoubleVar(value=60.0),
            "show_diagnostic_map": tk.BooleanVar(value=False),
            "selected_diagnostic_map": tk.StringVar(),
            # --- Geostrophic Wind Variables ---
            "enable_geostrophic_winds": tk.BooleanVar(value=True),
            "geostrophic_wind_kmh": tk.DoubleVar(value=40.0),
            "geostrophic_blur_sigmas": tk.StringVar(value="2, 4, 8"),
            "geostrophic_advection_steps": tk.IntVar(value=4), # Changed default to 2
            "world_diameter_km": tk.IntVar(value=12000), # Changed default to 12000
            "final_temp_blur_sigma": tk.DoubleVar(value=3.0), # Added final temperature blur sigma
            # --- Altitude Tint Variables ---
            "enable_altitude_tint": tk.BooleanVar(value=False),
            # --- Formula Variables ---
            "heightmap_formula": tk.StringVar(value=DEFAULT_FORMULAS["heightmap_formula"]),
            "temperature_base_k_formula": tk.StringVar(value=DEFAULT_FORMULAS["temperature_base_k_formula"]),
            "temperature_greenhouse_k_formula": tk.StringVar(value=DEFAULT_FORMULAS["temperature_greenhouse_k_formula"]),
            "temperature_latitude_gradient_formula": tk.StringVar(value=DEFAULT_FORMULAS["temperature_latitude_gradient_formula"]),
            "temperature_lapse_rate_term_formula": tk.StringVar(value=DEFAULT_FORMULAS["temperature_lapse_rate_term_formula"]),
            "rainfall_orographic_effect_formula": tk.StringVar(value=DEFAULT_FORMULAS["rainfall_orographic_effect_formula"]),
        
        }

        self.scaling_manager = ScalingManager(self)
        self.history = HistoryManager(self)
        self.biomes = []

        # --- LAYER VISIBILITY VARIABLES ---
        self.layer_vars = {
            "Biomes": tk.BooleanVar(value=True),
            "Rivers": tk.BooleanVar(value=True),
            "Temperature": tk.BooleanVar(value=False),
            "Rainfall": tk.BooleanVar(value=False),
            "Heightmap": tk.BooleanVar(value=False),
            "Tectonic Plates": tk.BooleanVar(value=False),
            "Altitude Tint": tk.BooleanVar(value=True), # This key MUST be present here
        }
        # --- END OF LAYER VISIBILITY VARIABLES ---

        self.paint_vars = {
            "paint_mode": tk.StringVar(value="Terrain"),
            "brush_tool": tk.StringVar(value="Raise"),
            "brush_shape": tk.StringVar(value="Circle"),
            "brush_size": tk.IntVar(value=30),
            "brush_strength": tk.DoubleVar(value=0.05),
            "level_height": tk.DoubleVar(value=0.5),
            "biome_to_paint": tk.StringVar(),
        }
        self.last_gen_data = {}
        self.generation_queue = queue.Queue()
        self.current_filepath = None
        self.paint_action_data = None
        self.erosion_widgets = []
        self.real_value_controls = []
        self.diagnostic_images = {}
        self.formula_widgets = {}

        self.create_menu()
        main_frame = ttk.Frame(root)
        main_frame.grid(row=1, column=0, sticky="nsew")
        main_pane = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_notebook = ttk.Notebook(main_pane)
        main_pane.add(left_notebook, weight=1)
        self.properties_frame = ttk.Frame(left_notebook, padding=10)
        left_notebook.add(self.properties_frame, text=" Properties ")
        self.scale_frame = ttk.Frame(left_notebook, padding=10)
        left_notebook.add(self.scale_frame, text=" Scale ")
        self.paint_frame = ttk.Frame(left_notebook, padding=10)
        left_notebook.add(self.paint_frame, text=" Paint ")
        self.realism_frame = ttk.Frame(left_notebook, padding=10)
        left_notebook.add(self.realism_frame, text=" Realism ")
        self.formulas_frame = ttk.Frame(left_notebook, padding=10)
        left_notebook.add(self.formulas_frame, text=" Formulas ")
        self.diagnostics_frame = ttk.Frame(left_notebook, padding=10)
        left_notebook.add(self.diagnostics_frame, text=" Diagnostics ")


        canvas_frame = ttk.Frame(main_pane, style="Map.TFrame")
        main_pane.add(canvas_frame, weight=4)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.map_canvas = ZoomableCanvas(canvas_frame, background="#282828", highlightthickness=0)
        self.map_canvas.grid(row=0, column=0, sticky="nsew")
        self.map_canvas.set_offset_update_callback(lambda r: self.scroll_x.set(r))
        self.scroll_x = tk.DoubleVar(value=0.0)
        self.scrollbar_x = ttk.Scale(canvas_frame, orient="horizontal", variable=self.scroll_x,
                                     command=lambda v: self.map_canvas.set_x_offset_ratio(float(v)))
        self.scrollbar_x.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(1, weight=0)

        right_notebook = ttk.Notebook(main_pane)
        main_pane.add(right_notebook, weight=1)
        self.layers_frame = ttk.Frame(right_notebook, padding=10)
        right_notebook.add(self.layers_frame, text=" Layers ")
        self.biome_frame = ttk.Frame(right_notebook, padding=10)
        right_notebook.add(self.biome_frame, text=" Biomes ")
        self.stats_frame = ttk.Frame(right_notebook, padding=10)
        right_notebook.add(self.stats_frame, text=" Stats ")

        self.status_var = tk.StringVar(value="Welcome! Open a project or generate a new world.")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5).grid(row=2, column=0, sticky="ew")

        # --- Create UI Sections ---
        self.create_properties_controls()
        self.create_scale_controls()
        # self.create_paint_controls() # Called later as it depends on update_biome_paint_list
        self.create_layers_controls()
        # self.create_biome_editor() # Called later as it depends on update_biome_paint_list, load_biomes_to_tree
        self.create_stats_panel()
        self.create_realism_controls()
        self.create_formulas_panel()
        # self.create_diagnostics_controls() # Called later as it depends on _update_diagnostic_list
        # --- End Create UI Sections ---

        self.is_painting = False
        self.brush_preview_id = self.map_canvas.create_oval(0, 0, 0, 0, outline="#cccccc", dash=(4, 2), state="hidden")
        self.map_canvas.bind("<Configure>", self._on_canvas_configure)
        self.map_canvas.bind("<Motion>", self._on_mouse_move)
        self.map_canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.map_canvas.bind("<B2-Motion>", self.on_pan_move)
        self.map_canvas.bind("<MouseWheel>", self.on_zoom)
        self.map_canvas.bind("<ButtonPress-1>", self.on_paint_start)
        self.map_canvas.bind("<B1-Motion>", self.on_paint_move)
        self.map_canvas.bind("<ButtonRelease-1>", self.on_paint_end)
        self._update_erosion_controls_state() # Depends on erosion_widgets being populated later

        # --- Call UI sections that depend on helper methods being defined ---
        self.create_biome_editor() # Calls load_biomes_to_tree -> update_biome_paint_list
        self.create_paint_controls() # Calls _on_paint_mode_change -> update_biome_paint_list
        self.create_diagnostics_controls() # Calls _update_diagnostic_list
        # --- End Dependent UI Section Calls ---


    # --- Helper methods used by UI creation or early calls ---
    def update_biome_paint_list(self):
        """Updates the list of biomes available in the paint tool combobox."""
        # This needs self.biomes and self.biome_paint_cb (from create_paint_controls)
        # Ensure create_paint_controls is called before this is triggered by _on_paint_mode_change
        try:
            biomes = [b['name'] for b in self.get_biomes_from_tree() if not b.get('is_water')]
            if hasattr(self, 'biome_paint_cb') and self.biome_paint_cb is not None:
                self.biome_paint_cb["values"] = biomes
                if biomes:
                    # Check if the currently selected biome is still valid, if not, set the first one
                    current_biome = self.paint_vars["biome_to_paint"].get()
                    if current_biome not in biomes and biomes:
                        self.paint_vars["biome_to_paint"].set(biomes[0])
                    elif not biomes:
                        self.paint_vars["biome_to_paint"].set("") # Clear if no biomes
        except Exception as e:
            print(f"Error in update_biome_paint_list: {e}")
            return


    def _on_paint_mode_change(self, event=None):
        """Switches visibility between Terrain and Biome paint controls."""
        # Needs self.terrain_tools_frame and self.biome_tools_frame (from create_paint_controls)
        # Needs update_biome_paint_list
        mode = self.paint_vars["paint_mode"].get()
        if mode == "Terrain":
            if hasattr(self, 'biome_tools_frame'): self.biome_tools_frame.grid_remove()
            if hasattr(self, 'terrain_tools_frame'): self.terrain_tools_frame.grid()
        else: # Mode is "Biome"
            if hasattr(self, 'terrain_tools_frame'): self.terrain_tools_frame.grid_remove()
            if hasattr(self, 'biome_tools_frame'): self.biome_tools_frame.grid()
            self.update_biome_paint_list() # This call caused the error previously


    def _update_diagnostic_list(self):
        """Populates the diagnostic map combobox."""
        # Needs self.last_gen_data and self.diag_cb, self.diag_map_cb (from create_diagnostics_controls)
        # Needs gen.create_*_image functions
        # Needs self.diagnostic_images dict
        diag_maps_data = self.last_gen_data.get("diagnostic_maps", {})
        if not diag_maps_data:
            if hasattr(self, 'diag_cb'): self.diag_cb.config(state="disabled")
            if hasattr(self, 'diag_map_cb'): self.diag_map_cb.config(state="disabled", values=[]) # Clear values
            self.vars["show_diagnostic_map"].set(False)
            self.diagnostic_images.clear() # Clear stored images
            return

        self.diagnostic_images.clear()
        map_names = []
        # Sort diagnostic maps for consistent ordering in the combobox
        for name, data_array in sorted(diag_maps_data.items()):
            if data_array is None or data_array.size == 0: # Skip empty or None diagnostic maps
                 continue
            try:
                img = None
                # Determine image type based on name hints or shape
                if "angles" in name:
                    # Angle maps typically stored in radians
                    img = gen.create_angle_map_image(data_array)
                elif "temp" in name:
                    # Temperature maps typically stored in Kelvin
                    img = gen.create_temperature_image(data_array)
                elif "rain" in name or "orographic" in name or "moisture" in name or "deposition" in name:
                    # Rainfall/moisture related maps
                    # Raw deposition might need normalization before coloring
                    if "raw" in name or "deposition" in name and data_array.max() > 1e-9: # Normalize if it looks like raw deposition
                         img = gen.create_rainfall_image(gen.normalize_map(data_array) * 2000) # Arbitrary scale for visualization
                    else: # Otherwise assume it's rainfall in mm/yr or similar
                         img = gen.create_rainfall_image(data_array)
                else: # Default heatmap visualization for other maps (e.g., tectonic, noise, erosion)
                    img = gen.create_heatmap_image(data_array)

                if img:
                    self.diagnostic_images[name] = img
                    map_names.append(name)
                else:
                    print(f"Warning: Could not create image for diagnostic map '{name}'. Skipping.")
            except Exception as e:
                print(f"Error creating image for diagnostic map '{name}': {e}")
                # Continue to the next diagnostic map


        if hasattr(self, 'diag_map_cb'): self.diag_map_cb["values"] = map_names
        if hasattr(self, 'diag_cb'): self.diag_cb.config(state="normal")

        if map_names:
             if hasattr(self, 'diag_map_cb'): self.diag_map_cb.config(state="readonly")
             current_selection = self.vars["selected_diagnostic_map"].get()
             # If current selection is invalid or empty, set the first one
             if not current_selection or current_selection not in map_names:
                  self.vars["selected_diagnostic_map"].set(map_names[0])
        else:
             # If no diagnostic maps were successfully processed
             if hasattr(self, 'diag_cb'): self.diag_cb.config(state="disabled")
             if hasattr(self, 'diag_map_cb'): self.diag_map_cb.config(state="disabled", values=[])
             self.vars["show_diagnostic_map"].set(False)
             self.diagnostic_images.clear()
             self.status_var.set("No diagnostic maps available.") # Update status bar


        # If showing diagnostic map is enabled, update the display to show it
        if self.vars["show_diagnostic_map"].get() and map_names:
             self.update_display() # This will call update_display with the current diagnostic setting


    # --- End Helper methods ---
    def create_layers_controls(self):
        f = self.layers_frame
        f.columnconfigure(0, weight=1)
        lf_vis = ttk.Labelframe(f, text="Layer Visibility", padding=5)
        lf_vis.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for n, v in self.layer_vars.items():
            ttk.Checkbutton(lf_vis, text=n, variable=v, command=self.update_display).pack(anchor="w", padx=5)
        lf_riv = ttk.Labelframe(f, text="River Carving", padding=5)
        lf_riv.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        lf_riv.columnconfigure(1, weight=1)
        self.create_control_slider(lf_riv, "Strength Thresh:", self.vars["river_strength_threshold"], 0.0, 1.0, "Only carves rivers stronger than this value.")
        erosion_cb = ttk.Checkbutton(lf_riv, text="Enable Valley Erosion", variable=self.vars["enable_erosion"], command=self._update_erosion_controls_state)
        erosion_cb.grid(row=lf_riv.grid_size()[1], column=0, columnspan=3, sticky="w", padx=5, pady=5)
        Tooltip(erosion_cb, "If enabled, carves smooth valleys around rivers instead of just channels.")
        l1, s1, e1 = self.create_real_value_slider(lf_riv, "Erosion H-Thresh", self.vars["erosion_height_threshold"], "Valley erosion only affects terrain higher than this value.")
        l2, s2, e2 = self.create_control_slider(lf_riv, "Erosion Sharpness:", self.vars["erosion_sharpness"], 1.0, 20.0, "How sharp the river valley walls are. Higher value = steeper.")
        self.erosion_widgets.extend([l1, s1, e1, l2, s2, e2])
        impl_button = ttk.Button(lf_riv, text="Carve Rivers into Heightmap", command=self.implement_rivers, style="Accent.TButton")
        impl_button.grid(row=lf_riv.grid_size()[1], column=0, columnspan=3, sticky="ew", pady=10, ipady=3)

        # Altitude Tint section
        lf_tint = ttk.Labelframe(f, text="Altitude Tint", padding=5)
        lf_tint.grid(row=lf_riv.grid_size()[1] + 1, column=0, sticky="ew", pady=(0, 10))
        lf_tint.columnconfigure(1, weight=1)

        tint_cb = ttk.Checkbutton(lf_tint, text="Enable Altitude Tint", variable=self.vars["enable_altitude_tint"], command=self.update_display)
        tint_cb.grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=(5, 2))
        Tooltip(tint_cb, "Applies a tint to land based on its height above sea level.")
        # The slider for tint strength was removed in a previous step as it's no longer needed

    def create_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="New World", command=self.run_generation_thread, accelerator="Ctrl+N")
        filemenu.add_command(label="Open Project...", command=self.open_project, accelerator="Ctrl+O")
        filemenu.add_separator()
        filemenu.add_command(label="Save", command=self.save_project, accelerator="Ctrl+S")
        filemenu.add_command(label="Save As...", command=self.save_as_project, accelerator="Ctrl+Shift+S")
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        self.editmenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=self.editmenu)
        self.editmenu.add_command(label="Undo", command=self.history.undo, accelerator="Ctrl+Z", state="disabled")
        self.editmenu.add_command(label="Redo", command=self.history.redo, accelerator="Ctrl+Y", state="disabled")
        self.editmenu.add_separator()
        self.editmenu.add_command(label="Clear All Biome Overrides", command=self.clear_biome_overrides, state="disabled")
        self.root.config(menu=menubar)
        self.root.bind_all("<Control-n>", lambda e: self.run_generation_thread())
        self.root.bind_all("<Control-o>", lambda e: self.open_project())
        self.root.bind_all("<Control-s>", lambda e: self.save_project())
        self.root.bind_all("<Control-Shift-S>", lambda e: self.save_as_project())
        self.root.bind_all("<Control-z>", lambda e: self.history.undo())
        self.root.bind_all("<Control-y>", lambda e: self.history.redo())

    # --- Most UI and Event methods ---
    # ... (on_pan_start, on_paint_move, etc.) ...
    def on_pan_start(self, e): self.pan_last_x, self.pan_last_y = e.x, e.y
    def on_pan_move(self, e):
        dx, dy = e.x - self.pan_last_x, e.y - self.pan_last_y
        self.map_canvas.pan(dx, dy)
        self.pan_last_x, self.pan_last_y = e.x, e.y
        self.scroll_x.set(self.map_canvas.get_x_offset_ratio())
    def on_zoom(self, e):
        self.map_canvas.zoom(e)
        self.scroll_x.set(self.map_canvas.get_x_offset_ratio())
    def on_paint_move(self, e):
        if self.is_painting:
            self._apply_brush(e)
            self._regenerate_visuals_from_data(full_recalc=False) # This calls update_display internally
            # self.update_display() # No need to call here, _regenerate_visuals_from_data calls update_display
            self._on_mouse_move(e) # Update status bar during painting
    def on_paint_start(self, e):
        if "heightmap" not in self.last_gen_data: return
        self.is_painting = True
        mode = self.paint_vars["paint_mode"].get()
        
        # Add debugging
        if mode == "Terrain":
            data_map = self.last_gen_data["heightmap"]
            if data_map is None:
                print("ERROR: heightmap is None!")
                self.is_painting = False
                return
        elif mode == "Biome":
            # Ensure biome_override_map exists
            if "biome_override_map" not in self.last_gen_data or self.last_gen_data["biome_override_map"] is None:
                h, w = self.last_gen_data["heightmap"].shape
                self.last_gen_data["biome_override_map"] = np.full((h, w), -1, dtype=np.int8)
            data_map = self.last_gen_data["biome_override_map"]
            if data_map is None:
                print("ERROR: biome_override_map is None!")
                self.is_painting = False
                return
        
        # Add this check before the copy
        if data_map is None:
            print("ERROR: data_map is None!")
            self.is_painting = False
            return
            
        self.paint_action_data = {"map": data_map, "before": data_map.copy(), "changed_indices": set()}


    def on_paint_end(self, e):
        if not self.is_painting: return
        self.is_painting = False # Stop painting state immediately

        if self.paint_action_data and self.paint_action_data["changed_indices"]:
            indices = tuple(np.array(list(self.paint_action_data["changed_indices"])).T)
            # Ensure before_values are taken from the *original* copy before any changes
            before_values = self.paint_action_data["before"][indices]
            # Ensure after_values are taken from the map *after* all brush strokes
            after_values = self.paint_action_data["map"][indices]

            # Only create history action if there was an actual change in values
            # This avoids cluttering history with brush strokes that didn't change anything (e.g., painting same biome)
            if not np.array_equal(before_values, after_values):
                action = PaintAction(self.paint_action_data["map"], indices, before_values, after_values)
                self.history.push(action)
                self.on_history_change() # This regenerates visuals and updates display/menu
            else:
                # If no effective change, still regenerate visuals to ensure display is correct after brush moves
                self._regenerate_visuals_from_data(full_recalc=False)
                self.update_display()
                self.update_edit_menu_state() # Update undo/redo state even if no change

        self.paint_action_data = None # Clear paint action data


    def on_history_change(self):
        """Callback after undo/redo action."""
        # This is triggered after history.push, history.undo, history.redo
        # The history action's apply/unapply method modifies the actual map data.
        # So we just need to regenerate visuals from the modified data.
        self._regenerate_visuals_from_data(full_recalc=False) # Regenerate visuals from current data state
        self.update_display() # Update the display
        self.update_edit_menu_state() # Update undo/redo menu state


    def clear_biome_overrides(self):
        """Clears all manually painted biome overrides."""
        # Needs self.last_gen_data["biome_override_map"]
        if "biome_override_map" in self.last_gen_data and self.last_gen_data["biome_override_map"] is not None:
            override_map = self.last_gen_data["biome_override_map"]
            if override_map is None:  # Double check
                self.status_var.set("Error: biome override map is None.")
                return
            map_before = override_map.copy()
            # Find all indices that are not the default (-1)
            changed_indices = np.where(map_before != -1)
            if changed_indices[0].size > 0:
                before_values = map_before[changed_indices]
                override_map.fill(-1) # Fill the map with the default -1
                after_values = override_map[changed_indices]
                # Record this as a single history action
                action = PaintAction(override_map, changed_indices, before_values, after_values)
                self.history.push(action)
                self.on_history_change() # Regenerate visuals and update display/menu
                self.status_var.set("All biome overrides have been cleared.")
            else:
                self.status_var.set("No biome overrides to clear.")
        else:
            messagebox.showwarning("No Data", "No biome override data available to clear.")
            self.status_var.set("No biome override data.")


    def _apply_brush(self, e):
        """Applies the active brush to the relevant map."""
        # Needs self.paint_vars, self.last_gen_data, self.biomes (from get_biomes_from_tree)
        # Modifies self.paint_action_data["map"] and tracks changes in "changed_indices"
        mode = self.paint_vars["paint_mode"].get()
        # paint_action_data["map"] is a reference to the actual map array
        data_map = self.paint_action_data["map"]

        # Ensure data_map is valid (e.g. not None)
        if data_map is None:
             print(f"Error: Cannot apply brush, data_map is None for mode '{mode}'.")
             self.is_painting = False # Stop painting
             return

        h, w = data_map.shape
        center_x, center_y = self.map_canvas.canvas_to_world(e.x, e.y)

        # Clamp brush center to map bounds for robustness
        center_x = np.clip(center_x, 0, w - 1)
        center_y = np.clip(center_y, 0, h - 1)

        size = self.paint_vars["brush_size"].get()
        # Generate brush mask coordinates relative to brush center
        y_coords, x_coords = np.ogrid[-size : size + 1, -size : size + 1]

        # Create brush shape mask
        shape = self.paint_vars["brush_shape"].get()
        if shape == "Circle": mask = x_coords**2 + y_coords**2 <= size**2
        elif shape == "Square": mask = np.ones((2 * size + 1, 2 * size + 1), dtype=bool)
        elif shape == "Diamond": mask = np.abs(x_coords) + np.abs(y_coords) <= size
        else:
             print(f"Warning: Unknown brush shape '{shape}'. Using Circle.")
             mask = x_coords**2 + y_coords**2 <= size**2 # Default to Circle

        # Calculate absolute world coordinates for the brush mask pixels
        # Add 0.5 for center-to-corner offset before casting to int
        world_y, world_x = np.where(mask)
        world_y_abs = (world_y + center_y - size + 0.5).astype(int)
        world_x_abs = (world_x + center_x - size + 0.5).astype(int)


        # Filter for indices that are within map bounds
        valid_indices_mask = (world_y_abs >= 0) & (world_y_abs < h) & (world_x_abs >= 0) & (world_x_abs < w)
        world_y_abs_valid = world_y_abs[valid_indices_mask]
        world_x_abs_valid = world_x_abs[valid_indices_mask]

        # Corresponding mask indices for the valid world coordinates
        brush_mask_y_valid = world_y[valid_indices_mask]
        brush_mask_x_valid = world_x[valid_indices_mask]

        if len(world_y_abs_valid) == 0:
             # print("Brush is completely outside map bounds.") # Could be noisy
             return # No pixels to paint

        # Add the valid absolute indices to the changed set for history tracking
        self.paint_action_data["changed_indices"].update(zip(world_y_abs_valid, world_x_abs_valid))

        # --- Apply painting based on mode ---
        if mode == "Terrain":
            strength = self.paint_vars["brush_strength"].get()
            tool = self.paint_vars["brush_tool"].get()

            # Calculate falloff based on distance from the brush center for the valid pixels
            # Use the original mask coordinates (relative to brush center) for distance calculation
            # --- CORRECTED LINE ---
            dist_sq = (brush_mask_x_valid - size) ** 2 + (brush_mask_y_valid - size) ** 2 # Distance from center of mask
             # --- END CORRECTED LINE ---
            falloff = np.clip(1 - dist_sq / (size**2), 0, 1) if size > 0 else np.ones_like(brush_mask_x_valid, dtype=np.float32) # Full strength if size is 0

            # Effective strength is modulated by falloff
            eff_strength = strength * falloff

            if tool == "Raise":
                data_map[world_y_abs_valid, world_x_abs_valid] += eff_strength
            elif tool == "Lower":
                data_map[world_y_abs_valid, world_x_abs_valid] -= eff_strength
            elif tool == "Level":
                target = self.paint_vars["level_height"].get()
                # Interpolate between current value and target based on effective strength
                data_map[world_y_abs_valid, world_x_abs_valid] = (
                    data_map[world_y_abs_valid, world_x_abs_valid] * (1 - eff_strength) +
                    target * eff_strength
                )
            elif tool == "Smooth":
                # Smoothing applies to a rectangular region encompassing the brush area
                # This is less precise than a masked smooth but simpler to implement with gaussian_filter
                # Find the bounding box of the brush on the map
                y_min, y_max = np.min(world_y_abs_valid), np.max(world_y_abs_valid)
                x_min, x_max = np.min(world_x_abs_valid), np.max(world_x_abs_valid)

                # Add a small buffer to ensure smoothing covers the edge
                buffer = int(size * 0.2) # Arbitrary buffer based on brush size
                y_min = max(0, y_min - buffer)
                y_max = min(h - 1, y_max + buffer)
                x_min = max(0, x_min - buffer)
                x_max = min(w - 1, x_max + buffer)

                # Apply Gaussian filter to the bounding box region
                # Use the 'strength' variable as the sigma for the blur
                if x_max > x_min and y_max > y_min:
                     # Ensure sigma is positive, default to small blur if strength is zero
                     sigma = max(0.5, strength * 10.0) # Scale strength to a reasonable sigma range
                     region_to_smooth = data_map[y_min : y_max + 1, x_min : x_max + 1]
                     # Apply filter and update the region in the data_map
                     data_map[y_min : y_max + 1, x_min : x_max + 1] = gaussian_filter(region_to_smooth, sigma=sigma, mode='wrap') # Use wrap mode


            # Clip terrain height to the valid 0.0 - 1.0 range
            np.clip(data_map, 0.0, 1.0, out=data_map)

        elif mode == "Biome":
            # Biome painting applies the selected biome index directly to pixels under the brush mask
            biome_name = self.paint_vars["biome_to_paint"].get()
            biomes = self.get_biomes_from_tree() # Get current biomes from the editor
            # Find the index of the selected biome
            biome_index = next((i for i, b in enumerate(biomes) if b["name"] == biome_name), -1) # -1 if not found

            if biome_index != -1:
                 # Apply the biome index to all valid pixels under the brush
                 data_map[world_y_abs_valid, world_x_abs_valid] = biome_index
            else:
                 print(f"Warning: Selected biome '{biome_name}' not found in current biome list. Cannot paint.")
                 # Do not paint if biome index is invalid


    def _on_canvas_configure(self, e):
        """Handles canvas resizing."""
        # Needs self.map_canvas, self.last_gen_data
        if "heightmap" in self.last_gen_data:
             self.map_canvas.redraw() # Redraw the map when canvas size changes
        else:
             self._show_initial_message() # Show initial message if no map is loaded


    def _show_initial_message(self):
        """Displays a message on the canvas when no world is loaded."""
        # Needs self.map_canvas
        self.map_canvas.delete("all") # Clear existing items
        canvas_w, canvas_h = self.map_canvas.winfo_width(), self.map_canvas.winfo_height()
        # Only show message if canvas has a size
        if canvas_w > 1 and canvas_h > 1:
            cx, cy = canvas_w / 2, canvas_h / 2
            self.map_canvas.create_text(cx, cy, text="Generate a new world or open a project.", fill="white", font=("Arial", 14), justify="center")


    def _on_mouse_move(self, e):
        """Updates status bar and brush preview on mouse motion over the canvas."""
        # Needs self.last_gen_data, self.map_canvas, self.paint_vars, self.scaling_manager, self.status_var
        # Needs self.brush_preview_id
        if "heightmap" in self.last_gen_data and self.last_gen_data["heightmap"] is not None:
            # Get world coordinates under the mouse pointer
            mx, my = self.map_canvas.canvas_to_world(e.x, e.y)

            # Update brush preview position
            radius = self.paint_vars["brush_size"].get() * self.map_canvas.zoom_level # Scale brush size by zoom
            # Center the oval on the mouse cursor
            self.map_canvas.coords(self.brush_preview_id, e.x - radius, e.y - radius, e.x + radius, e.y + radius)
            self.map_canvas.itemconfig(self.brush_preview_id, state="normal") # Make brush visible


            h, w = self.last_gen_data["heightmap"].shape
            # Check if mouse is within map bounds
            if 0 <= mx < w and 0 <= my < h:
                ix, iy = int(mx), int(my)
                norm_h = self.last_gen_data["heightmap"][iy, ix]
                sea_level_norm = self.vars["sea_level"].get()
                unit = self.scaling_manager.get_unit_suffix()

                # Display height/depth relative to sea level
                if norm_h > sea_level_norm:
                    # Land: Display height above sea level
                    display_val = self.scaling_manager.to_real(norm_h - sea_level_norm, above_sea=True)
                    label = "Height"
                else:
                    # Water: Display depth below sea level
                    display_val = self.scaling_manager.to_real(sea_level_norm - norm_h, above_sea=False)
                    label = "Depth"

                status_str = f"X: {ix}, Y: {iy} | {label}: {display_val:.1f} {unit}"

                # Add climate info if available
                if "temperature_map" in self.last_gen_data and self.last_gen_data["temperature_map"] is not None:
                    temp_k = self.last_gen_data["temperature_map"][iy, ix]
                    disp_temp, temp_unit = self.scaling_manager.to_display_temp(temp_k)
                    status_str += f" | Temp: {disp_temp:.1f}{temp_unit}"
                if "rainfall_map" in self.last_gen_data and self.last_gen_data["rainfall_map"] is not None:
                    rain = self.last_gen_data["rainfall_map"][iy, ix]
                    status_str += f" | Rain: {rain:.0f}mm"

                # Add biome info if biome indices map is available
                if "biome_indices_map" in self.last_gen_data and self.last_gen_data["biome_indices_map"] is not None:
                     biome_idx = self.last_gen_data["biome_indices_map"][iy, ix]
                     biomes = self.get_biomes_from_tree()
                     # Find the biome name by index
                     biome_name = "Unknown Biome"
                     # Ensure biome_idx is a valid index for the current biomes list
                     if 0 <= biome_idx < len(biomes):
                          biome_name = biomes[biome_idx]['name']

                     # Check for biome override
                     if "biome_override_map" in self.last_gen_data and self.last_gen_data["biome_override_map"] is not None:
                          override_idx = self.last_gen_data["biome_override_map"][iy, ix]
                          if override_idx != -1 and 0 <= override_idx < len(biomes):
                               override_name = biomes[override_idx]['name']
                               status_str += f" | Biome: {override_name} (Override)"
                          else:
                               status_str += f" | Biome: {biome_name}" # Use auto-classified biome if no override
                     else:
                          status_str += f" | Biome: {biome_name}" # No override map exists


                status_str += f" | Zoom: {self.map_canvas.zoom_level:.2f}x"
                self.status_var.set(status_str) # Update status bar text
            else:
                # Mouse is outside map bounds
                self.status_var.set(f"Zoom: {self.map_canvas.zoom_level:.2f}x") # Only show zoom
        else:
            # No heightmap data available
            self.map_canvas.itemconfig(self.brush_preview_id, state="hidden") # Hide brush
            self.status_var.set("Generate a new world or open a project.") # Default message


    def _on_unit_or_scale_change(self, event=None):
        """Updates UI elements and biome table headings when unit system or max height changes."""
        # Needs update_real_value_displays, load_biomes_to_tree, update_stats_panel
        self.update_real_value_displays()
        # Reload biomes to update temperature unit in biome treeview headings
        # Need to pass the *current* biomes data, not load defaults
        self.load_biomes_to_tree(self.get_biomes_from_tree()) # Use get_biomes_from_tree to get current list
        self.update_stats_panel() # Update stats panel which includes temp/rain units

    def update_real_value_displays(self):
        """Updates entry widgets that display real-world values."""
        # Needs self.real_value_controls, self.scaling_manager
        for norm_var, entry_var, label, suffix_func in self.real_value_controls:
            # Check if the var is a DoubleVar or IntVar before getting value
            try:
                norm_value = norm_var.get()
                # Check if the variable name corresponds to height/depth for real conversion
                # This is a bit fragile, relies on naming convention or knowing which vars are height-related
                # For now, assume all vars in real_value_controls are height-related 0-1 normalized values
                real_val = self.scaling_manager.to_real(norm_value, above_sea=True)
                entry_var.set(f"{real_val:.1f}")
            except (tk.TclError, ValueError) as e:
                # Handle cases where the variable might not be a valid number
                print(f"Error updating real value display for {norm_var}: {e}")
                entry_var.set("Error") # Display error in the entry

            if label:
                # Update label with the current unit suffix
                base_text = label.cget("text").split("(")[0].strip()
                label.config(text=f"{base_text} ({self.scaling_manager.get_unit_suffix()}):")


    def update_edit_menu_state(self):
        """Updates the enabled/disabled state of Undo/Redo menu items."""
        # Needs self.editmenu, self.history, self.last_gen_data
        if hasattr(self, 'editmenu'): # Defensive check
            self.editmenu.entryconfig("Undo", state="normal" if self.history.undo_stack else "disabled")
            self.editmenu.entryconfig("Redo", state="normal" if self.history.redo_stack else "disabled")
            # Enable Clear Biome Overrides only if there is map data and an override map might exist
            self_editmenu_clear_state = "disabled"
            if self.last_gen_data and "biome_override_map" in self.last_gen_data and self.last_gen_data["biome_override_map"] is not None:
                 # Check if there's anything to clear (any value != -1)
                 if np.any(self.last_gen_data["biome_override_map"] != -1):
                      self_editmenu_clear_state = "normal"

            self.editmenu.entryconfig("Clear All Biome Overrides", state=self_editmenu_clear_state)


    def _update_erosion_controls_state(self):
        """Updates the enabled/disabled state of erosion-related controls."""
        # Needs self.vars["enable_erosion"], self.erosion_widgets
        state = "normal" if self.vars["enable_erosion"].get() else "disabled"
        for widget in self.erosion_widgets:
             if widget.winfo_exists(): # Check if widget still exists
                 widget.config(state=state)

    # --- Standard Control Creation Helper ---
    def create_control_slider(self, p, l, v, f, t, vn, i=False):
        """Creates a label, slider, and entry for a control variable."""
        # Needs Tooltip
        r = p.grid_size()[1] # Get the next available row
        label = ttk.Label(p, text=l)
        label.grid(row=r, column=0, sticky="w", padx=5, pady=2)
        e = ttk.Entry(p, textvariable=v, width=8)
        e.grid(row=r, column=2, padx=5)
        # Define command for slider to update entry format
        slider_command = None
        if i: # If variable is integer
            slider_command = lambda val: v.set(int(float(val)))
        else: # If variable is float
            slider_command = lambda val: v.set(round(float(val), 4)) # Round to 4 decimal places

        s = ttk.Scale(p, from_=f, to=t, orient=tk.HORIZONTAL, variable=v, command=slider_command)
        s.grid(row=r, column=1, sticky="ew", padx=5)

        # Add tooltips
        Tooltip(label, vn); Tooltip(s, vn); Tooltip(e, vn)

        # Bind <Return> key in entry to update slider value
        def update_slider_from_entry(event=None):
             try:
                  # Attempt to convert entry value to the variable type (float or int)
                  if i:
                      val = int(v.get())
                  else:
                       val = float(v.get())

                  # Ensure value is within the slider's range before setting
                  val = max(f, min(val, t))
                  v.set(val) # Update the variable, which also updates the slider
             except (ValueError, tk.TclError):
                  # If entry value is invalid, revert entry to current variable value
                  # This also happens automatically because textvariable is bound
                  pass # Just pass, the entry will revert


        e.bind("<Return>", update_slider_from_entry)
        # Consider also binding <FocusOut> for entries
        # e.bind("<FocusOut>", update_slider_from_entry) # Optional, can be annoying


        return label, s, e # Return the widgets


    def create_real_value_slider(self, parent, label_text, norm_var, tooltip_text):
        """Creates a slider for a normalized 0-1 value linked to a real-world value entry."""
        # Needs self.scaling_manager, Tooltip
        # Appends to self.real_value_controls
        r = parent.grid_size()[1] # Get the next available row

        # Initial label text includes current unit suffix
        suffix = self.scaling_manager.get_unit_suffix()
        label = ttk.Label(parent, text=f"{label_text} ({suffix}):")
        label.grid(row=r, column=0, sticky="w", padx=5, pady=2)

        # Entry textvariable holds the real-world value string
        entry_var = tk.StringVar()
        entry = ttk.Entry(parent, textvariable=entry_var, width=8)
        entry.grid(row=r, column=2, padx=5)

        # Scale textvariable holds the normalized 0-1 value
        scale = ttk.Scale(parent, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=norm_var)
        scale.grid(row=r, column=1, sticky="ew", padx=5)

        # Command/Trace functions to keep scale and entry in sync
        def update_entry_from_scale(*args):
             try:
                  # Convert normalized value to real value using scaling manager
                  real_val = self.scaling_manager.to_real(norm_var.get(), above_sea=True)
                  entry_var.set(f"{real_val:.1f}") # Update entry text
             except (tk.TclError, ValueError) as e:
                  print(f"Error converting normalized value {norm_var.get()} to real: {e}")
                  entry_var.set("Error")

        def update_scale_from_entry(event=None):
             try:
                  # Get value from entry, convert to float
                  real_value = float(entry_var.get())
                  # Convert real value to normalized value using scaling manager
                  norm_value = self.scaling_manager.to_normalized(real_value, above_sea=True)
                  # Ensure normalized value is within 0-1 range before setting
                  norm_value = max(0.0, min(norm_value, 1.0))
                  norm_var.set(norm_value) # Update the normalized variable, updates slider
             except (ValueError, tk.TclError):
                  # If entry value is invalid, revert entry to value derived from current norm_var
                  update_entry_from_scale() # Call the other update function

        # Link the normalized variable to update the entry
        # Use 'write' trace mode to trigger whenever the variable's value changes
        norm_var.trace_add("write", update_entry_from_scale)

        # Link the entry to update the normalized variable on user input
        entry.bind("<Return>", update_scale_from_entry)
        entry.bind("<FocusOut>", update_scale_from_entry) # Update when entry loses focus

        # Initialize the entry display when the control is created
        update_entry_from_scale()

        # Store info for later updates (e.g., when unit system changes)
        self.real_value_controls.append((norm_var, entry_var, label, self.scaling_manager.get_unit_suffix))

        # Add tooltips
        Tooltip(label, tooltip_text); Tooltip(scale, tooltip_text); Tooltip(entry, tooltip_text)

        return label, scale, entry # Return the widgets


    # --- UI Creation Methods ---
    def create_scale_controls(self):
        # ... (existing code - unchanged) ...
        f = self.scale_frame
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="Unit System:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        cb = ttk.Combobox(f, textvariable=self.vars["unit_system"], values=["Metric", "Imperial"], state="readonly")
        cb.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        cb.bind("<<ComboboxSelected>>", self._on_unit_or_scale_change)
        max_h_var = self.vars["elevation_range_m"]
        label = ttk.Label(f, text="Elevation Range (m):")
        label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        entry = ttk.Entry(f, textvariable=max_h_var)
        entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        entry.bind("<Return>", self._on_unit_or_scale_change)
        entry.bind("<FocusOut>", self._on_unit_or_scale_change)
        Tooltip(label, "Total vertical span from deepest ocean to highest mountain in meters.")
        Tooltip(entry, "Total vertical span from deepest ocean to highest mountain in meters.")


    def create_properties_controls(self):
        # ... (existing code - unchanged) ...
        f = self.properties_frame
        f.columnconfigure(0, weight=1)
        lf_g = ttk.Labelframe(f, text="General", padding=5)
        lf_g.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        lf_g.columnconfigure(1, weight=1)
        f_s = ttk.Frame(lf_g)
        f_s.grid(row=0, column=0, columnspan=3, sticky="ew", pady=2)
        ttk.Label(f_s, text="Seed:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(f_s, textvariable=self.vars["seed"]).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(f_s, text="🎲", width=3, command=lambda: self.vars["seed"].set(random.randint(0, 10000))).pack(side=tk.LEFT, padx=(5, 0))
        self.create_control_slider(lf_g, "Map Size:", self.vars["size"], 128, 2048, "Map Size", True)
        lf_t = ttk.Labelframe(f, text="Terrain", padding=5)
        lf_t.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        lf_t.columnconfigure(1, weight=1)
        self.create_control_slider(lf_t, "Land Scale:", self.vars["continent_scale"], 50.0, 500.0, "Landmass Scale")
        self.create_control_slider(lf_t, "Flow Distort:", self.vars["flow_distortion"], 0, 200.0, "Flow Distortion")
        self.create_control_slider(lf_t, "Land Detail:", self.vars["continent_octaves"], 1, 10, "Landmass Detail", True)
        lf_tec = ttk.Labelframe(f, text="Tectonics", padding=5)
        lf_tec.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        lf_tec.columnconfigure(1, weight=1)
        self.create_control_slider(lf_tec, "Plate Count:", self.vars["plate_points"], 4, 50, "Plate Count", True)
        self.create_control_slider(lf_tec, "Uplift:", self.vars["uplift_magnitude"], 0.0, 2.0, "Tectonic Uplift")
        self.create_control_slider(lf_tec, "Smoothing:", self.vars["tectonic_smoothing"], 0.0, 10.0, "Tectonic Smoothing")
        self.create_control_slider(lf_tec, "Hypsometry:", self.vars["hypsometric_strength"], 0.5, 1.5, "Hypsometric Strength")
        self.create_control_slider(lf_tec, "Ridge Str:", self.vars["ridge_strength"], 0.0, 1.5, "Ridge Strength")
        self.create_control_slider(lf_tec, "Ridge Scale:", self.vars["ridge_scale"], 10.0, 200.0, "Ridge Scale")
        self.create_control_slider(lf_tec, "Ridge Detail:", self.vars["ridge_octaves"], 1, 8, "Ridge Detail", True)
        lf_r = ttk.Labelframe(f, text="Rivers / Flow", padding=5)
        lf_r.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        lf_r.columnconfigure(1, weight=1)
        self.create_control_slider(lf_r, "Particle Count:", self.vars["particle_count"], 1000, 20000, "Number of particles in the flow simulation.", True)
        self.create_control_slider(lf_r, "Particle Life:", self.vars["particle_steps"], 100, 2000, "Max number of steps each particle can take.", True)
        self.create_control_slider(lf_r, "Trail Fade:", self.vars["particle_fade"], 0.95, 0.999, "How slowly particle trails fade. Lower = shorter, bolder trails.")
        self.create_control_slider(lf_r, "Flow Scale:", self.vars["flow_scale"], 0.005, 0.05, "Scale of the base wind/flow patterns. Larger value = smaller swirls.")
        self.create_control_slider(lf_r, "Flow Detail:", self.vars["flow_octaves"], 1, 8, "Detail octaves for the base flow patterns.", True)
        recalc_button = ttk.Button(lf_r, text="Recalculate Rivers", command=self.recalculate_rivers_visuals)
        recalc_button.grid(row=lf_r.grid_size()[1], column=0, columnspan=3, sticky="ew", pady=(10, 0), ipady=3)
        Tooltip(recalc_button, "Recalculates the river visualization based on the current heightmap and flow parameters. Use after painting terrain.")
        self.generate_button = ttk.Button(f, text="Generate New World", command=self.run_generation_thread, style="Accent.TButton")
        self.generate_button.grid(row=4, column=0, sticky="ew", pady=15, ipady=5)
        self.root.style.configure("Accent.TButton", font="-weight bold")

    def recalculate_rivers_visuals(self):
        """Recalculates only the river visualization layer from existing data."""
        # Needs self.last_gen_data["river_deposition_map"], self.last_gen_data["land_mask"]
        # Calls gen.create_river_image
        # Updates self.last_gen_data["river_layer"]
        # Calls update_display
        if "river_deposition_map" not in self.last_gen_data or self.last_gen_data["river_deposition_map"] is None:
             messagebox.showerror("Error", "River deposition data not found. Please generate a world first or run climate simulation.")
             self.status_var.set("River visualization failed: data missing.")
             return

        self.status_var.set("Recalculating river visualization..."); self.root.update_idletasks()

        # Need land_mask to clip rivers. Generate it if it's missing.
        land_mask = self.last_gen_data.get("land_mask")
        if land_mask is None and "heightmap" in self.last_gen_data and self.last_gen_data["heightmap"] is not None:
             hmap = self.last_gen_data["heightmap"]
             sea_level_norm = self.vars["sea_level"].get()
             land_mask = hmap > sea_level_norm
             # Optionally store this generated land mask? Or just generate on the fly.
             # Let's generate on the fly if not present from climate sim.
             print("Warning: Land mask missing, generating from heightmap for river visualization.")
        elif land_mask is None:
             messagebox.showerror("Error", "Heightmap data missing. Cannot generate land mask for rivers.");
             self.status_var.set("River visualization failed: heightmap missing.")
             return

        try:
             # Create the river image using the stored deposition map and land mask
             self.last_gen_data["river_layer"] = gen.create_river_image(self.last_gen_data["river_deposition_map"], land_mask)
             self.update_display() # Update the display with the new river layer
             self.status_var.set("River visualization updated.")
        except Exception as e:
             messagebox.showerror("Error", f"An error occurred regenerating river visualization: {e}");
             self.status_var.set("River visualization failed.")
             print(f"Error regenerating river visual: {e}")


    def create_paint_controls(self):
        # ... (existing code - unchanged) ...
        f = self.paint_frame
        f.columnconfigure(0, weight=1)
        ttk.Label(f, text="Paint Mode:").grid(row=0, column=0, sticky="w")
        self.paint_mode_cb = ttk.Combobox(f, textvariable=self.paint_vars["paint_mode"], values=["Terrain", "Biome"], state="readonly")
        self.paint_mode_cb.grid(row=1, column=0, sticky="ew", pady=5)
        self.paint_mode_cb.bind("<<ComboboxSelected>>", lambda e: self._on_paint_mode_change())
        self.terrain_tools_frame = ttk.Labelframe(f, text="Terrain Tools", padding=5)
        self.terrain_tools_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        ttk.Combobox(self.terrain_tools_frame, textvariable=self.paint_vars["brush_tool"], values=["Raise", "Lower", "Smooth", "Level"], state="readonly").pack(fill="x", pady=2)
        self.biome_tools_frame = ttk.Labelframe(f, text="Biome Tools", padding=5)
        self.biome_paint_cb = ttk.Combobox(self.biome_tools_frame, textvariable=self.paint_vars["biome_to_paint"], state="readonly")
        self.biome_paint_cb.pack(fill="x", pady=2)
        lf_brush = ttk.Labelframe(f, text="Brush Settings", padding=5)
        lf_brush.grid(row=3, column=0, sticky="nsew", pady=5)
        lf_brush.columnconfigure(1, weight=1)
        ttk.Combobox(lf_brush, textvariable=self.paint_vars["brush_shape"], values=["Circle", "Square", "Diamond"], state="readonly").grid(row=0, column=0, columnspan=3, sticky="ew", pady=2)
        self.create_control_slider(lf_brush, "Size:", self.paint_vars["brush_size"], 5, 200, "Brush Size", True)
        self.create_control_slider(lf_brush, "Strength:", self.paint_vars["brush_strength"], 0.01, 0.5, "Brush Strength")
        self.create_real_value_slider(lf_brush, "Level To", self.paint_vars["level_height"], "The target height for the Level brush.")
        self._on_paint_mode_change() # Call the handler once to set initial state

    def create_realism_controls(self):
        # ... (existing code - unchanged) ...
        f = self.realism_frame
        f.columnconfigure(0, weight=1)

        lf_h = ttk.Labelframe(f, text="Hydrology & Erosion", padding=5)
        lf_h.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        lf_h.columnconfigure(1, weight=1)
        self.create_control_slider(lf_h, "Sea Level:", self.vars["sea_level"], 0.0, 1.0, "Normalized sea level. Determines coastlines.")
        self.create_control_slider(lf_h, "Erodibility Scale:", self.vars["erodibility_scale"], 10.0, 150.0, "Scale of the Perlin noise controlling soil/rock erodibility. Larger value = smaller features.")
        r = lf_h.grid_size()[1]
        label = ttk.Label(lf_h, text="Max Erosion Depth (m):")
        label.grid(row=r, column=0, sticky="w", padx=5, pady=2)
        entry = ttk.Entry(lf_h, textvariable=self.vars["max_erosion_depth_m"], width=8)
        entry.grid(row=r, column=1, sticky="ew", padx=5)
        Tooltip(label, "The maximum erosion depth in meters, a scaling factor in the erosion formula.")
        Tooltip(entry, "The maximum erosion depth in meters, a scaling factor in the erosion formula.")
        ttk.Button(f, text="Apply Hydraulic Erosion", command=self.apply_hydraulic_erosion, style="Accent.TButton").grid(row=1, column=0, sticky="ew", pady=(0, 15), ipady=5)

        lf_c = ttk.Labelframe(f, text="Climate", padding=5)
        lf_c.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        lf_c.columnconfigure(1, weight=1)
        self.create_control_slider(lf_c, "Solar Intensity (W/m²):", self.vars["solar_intensity"], 1000, 1800, "Insolation at the top of the atmosphere.", True)
        self.create_control_slider(lf_c, "CO2 Level (ppm):", self.vars["co2_level_ppm"], 0, 2000, "Atmospheric CO2 concentration.", True)
        self.create_control_slider(lf_c, "Global Rainfall (mm/yr):", self.vars["global_rainfall"], 100, 4000, "Global average rainfall, affects greenhouse effect and erosion.", True)
        self.create_control_slider(lf_c, "Eq-Pole Temp Diff (°C):", self.vars["equator_pole_temp_diff_c"], 10.0, 80.0, "The temperature difference between the equator and the poles. Controls the climate gradient.")
        self.create_control_slider(lf_c, "Lapse Rate (°C/1km):", self.vars["lapse_rate_c_per_1000m"], 3.0, 12.0, "Rate at which temperature decreases with altitude.")
        self.create_control_slider(lf_c, "North Latitude:", self.vars["latitude_north_pole"], -90.0, 90.0, "Latitude of the top of the map.")
        self.create_control_slider(lf_c, "South Latitude:", self.vars["latitude_south_pole"], -90.0, 90.0, "Latitude of the bottom of the map.")
        self.create_control_slider(lf_c, "Base Rainfall (mm/yr):", self.vars["base_latitude_rain"], 0, 4000, "Base rainfall at the equator.", True)
        self.create_control_slider(lf_c, "Orographic Strength:", self.vars["orographic_strength"], 0.0, 5.0, "How strongly mountains affect rainfall (rain shadows).")
        self.create_control_slider(lf_c, "Rain Shadow Factor:", self.vars["rain_shadow_strength_factor"], 0.0, 5.0, "Multiplier for the strength of the rain shadow effect.")
        self.create_control_slider(lf_c, "Climate Smoothing:", self.vars["climate_smoothing_sigma"], 0.0, 20.0, "Simulates heat transport via oceans/atmosphere. Higher values create smoother, more gradual temperature zones.")
        self.create_control_slider(lf_c, "Final Temp Blur:", self.vars["final_temp_blur_sigma"], 0.0, 10.0, "Applies a final Gaussian blur to the temperature map before biome classification.")
        r = lf_c.grid_size()[1]

        ttk.Separator(lf_c).grid(row=r, column=0, columnspan=3, sticky="ew", pady=5)

        wind_cb = ttk.Checkbutton(lf_c, text="Enable Geostrophic Winds", variable=self.vars["enable_geostrophic_winds"])
        wind_cb.grid(row=r+1, column=0, columnspan=3, sticky="w", padx=5)
        Tooltip(wind_cb, "Simulates heat redistribution via large-scale winds flowing along temperature contours.")

        self.create_control_slider(lf_c, "World Diameter (km):", self.vars["world_diameter_km"], 500, 20000, "The diameter of the world in kilometers, used for wind speed scaling.", True)
        self.create_control_slider(lf_c, "Wind Speed (km/h):", self.vars["geostrophic_wind_kmh"], 5.0, 200.0, "Average speed of geostrophic winds.")
        self.create_control_slider(lf_c, "Advection Steps:", self.vars["geostrophic_advection_steps"], 1, 10, "Number of simulation steps for wind advection. More steps = more pronounced effect.", True)

        r = lf_c.grid_size()[1]
        ttk.Label(lf_c, text="Blur Sigmas:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        blur_entry = ttk.Entry(lf_c, textvariable=self.vars["geostrophic_blur_sigmas"])
        blur_entry.grid(row=r, column=1, columnspan=2, sticky="ew", padx=5, pady=2)
        Tooltip(blur_entry, "Comma-separated list of Gaussian blur sigmas (e.g., '2, 4, 8') to create multi-scale wind eddies.")


        ttk.Button(f, text="Apply Climate & Biome Simulation", command=self.apply_climate_simulation, style="Accent.TButton").grid(row=3, column=0, sticky="ew", pady=(0, 15), ipady=5)

    def apply_hydraulic_erosion(self):
        # ... (existing code - unchanged) ...
        if "heightmap" not in self.last_gen_data: messagebox.showerror("Error", "Please generate a world first."); return
        self.status_var.set("Applying hydraulic erosion..."); self.root.update_idletasks()
        hmap = self.last_gen_data["heightmap"]
        h, w = hmap.shape
        hmap_before = hmap.copy()
        diag_maps = self.last_gen_data.setdefault("diagnostic_maps", {})
        try:
            erodibility_scale, erodibility_octaves = self.vars["erodibility_scale"].get(), self.vars["erodibility_octaves"].get()
            max_erosion_depth_m = self.vars["max_erosion_depth_m"].get()
            seed = self.vars["seed"].get()
            rainfall_term = (self.vars["global_rainfall"].get() / 1000.0) ** 0.5
            dy, dx = np.gradient(hmap)
            gradient_magnitude = np.sqrt(dx**2 + dy**2)
            diag_maps["erosion_slope"] = gradient_magnitude
            gradient_term = np.clip(gradient_magnitude, 0, 2.0) ** 0.7
            erodibility_map = gen.generate_perlin_noise(w, h, erodibility_scale, erodibility_octaves, seed + 999)
            diag_maps["erosion_erodibility"] = erodibility_map
            max_erosion_depth_norm = self.scaling_manager.to_normalized(max_erosion_depth_m, above_sea=True)
            erosion_map = rainfall_term * gradient_term * erodibility_map * max_erosion_depth_norm
            diag_maps["erosion_amount"] = erosion_map
            hmap -= erosion_map
            np.clip(hmap, 0.0, 1.0, out=hmap)
            self._update_diagnostic_list() # Calls _update_diagnostic_list
            changed_indices = np.where(hmap != hmap_before)
            if changed_indices[0].size > 0:
                action = PaintAction(hmap, changed_indices, hmap_before[changed_indices], hmap[changed_indices])
                self.history.push(action)
                self.on_history_change() # This calls _regenerate_visuals_from_data -> update_display
                self.status_var.set("Hydraulic erosion complete. Use Ctrl+Z to undo.")
            else:
                 # If no change, still update diagnostic list and status
                 self._update_diagnostic_list()
                 self.status_var.set("No changes made. Adjust erosion parameters and try again.")
        except tk.TclError as e: messagebox.showerror("Invalid Input", f"Invalid setting for erosion: {e}"); self.status_var.set("Erosion failed due to invalid input.")
        except Exception as e: messagebox.showerror("Error", f"An error occurred during erosion: {e}"); self.status_var.set("Erosion failed.")


    def apply_climate_simulation(self):
        # Check if base data is available
        if "heightmap" not in self.last_gen_data or self.last_gen_data["heightmap"] is None or \
           "flow_angles" not in self.last_gen_data or self.last_gen_data["flow_angles"] is None or \
           "river_deposition_map" not in self.last_gen_data or self.last_gen_data["river_deposition_map"] is None:
             messagebox.showerror("Error", "Base world data (heightmap, flow, rivers) not found. Please generate a world first.")
             self.status_var.set("Climate simulation failed: base data missing.")
             return

        try:
            params = {k: v.get() for k, v in self.vars.items()}
            self._get_formulas_from_ui() # Get latest formulas from text widgets
            params.update({k: v.get() for k,v in self.vars.items() if '_formula' in k})

            biomes = self.get_biomes_from_tree()
            hmap = self.last_gen_data["heightmap"]
            flow_angles = self.last_gen_data["flow_angles"] # Get flow angles from stored data
            river_deposition_map = self.last_gen_data["river_deposition_map"] # Get river deposition from stored data


            self.status_var.set("Simulating climate (1/5): Calculating Albedo..."); self.root.update_idletasks()
            # Albedo calculation needs biome indices map if available, otherwise uses hmap
            if "biome_indices_map" in self.last_gen_data and self.last_gen_data["biome_indices_map"] is not None:
                current_albedo = gen.calculate_albedo_map(self.last_gen_data['biome_indices_map'], biomes, hmap, params.get('sea_level', SEA_LEVEL))
            else: # Otherwise, use a simple land/sea albedo
                current_albedo = gen.calculate_albedo_map(np.full_like(hmap, -1, dtype=np.int8), biomes, hmap, params.get('sea_level', SEA_LEVEL)) # Pass dummy indices map

            self.last_gen_data.setdefault("diagnostic_maps", {})["climate_albedo"] = current_albedo.copy() # Store albedo as diagnostic


            self.status_var.set("Simulating climate (2/5): Calculating Temperature..."); self.root.update_idletasks()
            # Generate Temperature Map (incl. base, greenhouse, latitude, lapse rate)
            temp_data = gen.generate_temperature_map(hmap, current_albedo, params, self.scaling_manager)
            self.last_gen_data.update(temp_data) # Updates 'temperature_map' and adds 'climate_base_temp_k' diagnostic

            # Step 3: Apply Geostrophic Wind Advection to Temperature (Conditional)
            if params.get('enable_geostrophic_winds', False):
                self.status_var.set("Simulating climate (3/5): Applying Geostrophic Winds..."); self.root.update_idletasks()
                # apply_geostrophic_wind_advection modifies the temperature map in place or returns new one
                advected_temp_data = gen.apply_geostrophic_wind_advection(self.last_gen_data['temperature_map'].copy(), params) # Pass a copy
                self.last_gen_data['temperature_map'] = advected_temp_data['temperature_map'] # Update the main temp map with the advected result
                self.last_gen_data.setdefault("diagnostic_maps", {}).update(advected_temp_data.get("diagnostics", {})) # Add diagnostics

            # Step 3.5: Apply Final Temperature Blur (After Advection)
            final_blur_sigma = params.get('final_temp_blur_sigma', 0)
            if final_blur_sigma > 0 and self.last_gen_data.get('temperature_map') is not None:
                 self.status_var.set("Simulating climate (3.5/5): Applying Final Temperature Blur..."); self.root.update_idletasks()
                 try:
                      # Apply Gaussian blur directly to the main temperature map
                      self.last_gen_data['temperature_map'] = gaussian_filter(self.last_gen_data['temperature_map'], sigma=final_blur_sigma, mode='wrap')
                      # Optionally add the blurred map as a diagnostic layer
                      self.last_gen_data.setdefault("diagnostic_maps", {})["climate_temp_final_blurred"] = self.last_gen_data['temperature_map'].copy()
                 except Exception as e:
                      print(f"Error applying final temperature blur: {e}")
                      # Continue without blur if it fails

            # Recalculate river deposition with temperature constraints
            self.status_var.set("Simulating climate (3.7/5): Updating Rivers..."); self.root.update_idletasks()
            river_deposition_data = gen.calculate_river_deposition(
                 hmap,
                 flow_angles,
                 params,
                 self.scaling_manager,
                 self.last_gen_data['temperature_map']
            )
            river_deposition_map = river_deposition_data['deposition_map']
            self.last_gen_data['river_deposition_map'] = river_deposition_map
            self.last_gen_data.setdefault("diagnostic_maps", {}).update(river_deposition_data.get("diagnostics", {}))


            # Step 4: Calculate Rainfall Map (NEW LOGIC)
            self.status_var.set("Simulating climate (4/5): Calculating Rainfall..."); self.root.update_idletasks()
            # Call the new rainfall function with all required inputs
            rainfall_data = gen.generate_rainfall_map(
                 hmap,
                 self.last_gen_data['temperature_map'], # Use the (potentially advected/blurred) temp map
                 flow_angles, # Use the base flow angles from generate_world_data
                 river_deposition_map, # Use the river deposition map from generate_world_data
                 params,
                 self.scaling_manager # Need scaling manager for KM conversion in rainfall
            )
            self.last_gen_data.update(rainfall_data) # Updates 'rainfall_map' and adds new rainfall diagnostics


            # Step 5: Final Biome Classification
            self.status_var.set("Simulating climate (5/5): Final Biome Classification..."); self.root.update_idletasks()
            # Biome classification uses the *final* temperature and rainfall maps
            classification_data = gen.generate_classification_maps(hmap, self.last_gen_data['temperature_map'], self.last_gen_data['rainfall_map'], biomes, params)
            self.last_gen_data.update(classification_data) # Updates biome_image, biome_indices_map, land_mask


            # Create visual layers based on the simulation results
            # Temperature and Rainfall images are created here based on the final maps
            self.last_gen_data["temperature_image"] = gen.create_temperature_image(self.last_gen_data["temperature_map"])
            self.last_gen_data["rainfall_image"] = gen.create_rainfall_image(self.last_gen_data["rainfall_map"])

            # Create River Image based on the deposition map (now done in _regenerate_visuals_from_data)
            # Altitude Tint is applied based on the biome image (now done in _regenerate_visuals_from_data)


            self._update_diagnostic_list() # Update the list of available diagnostic maps in the UI
            # The update_display and update_stats_panel calls happen after run_climate_thread returns in check_generation_queue
            # self.update_display()
            # self.update_stats_panel()
            self.status_var.set("Climate and biome simulation complete.") # Update status bar


        except (tk.TclError, ValueError) as e:
             messagebox.showerror("Invalid Input", f"Invalid setting for climate/formula: {e}"); self.status_var.set("Climate simulation failed due to invalid input."); import traceback; traceback.print_exc()
        except Exception as e:
             messagebox.showerror("Error", f"An error occurred during climate simulation: {e}"); self.status_var.set("Climate simulation failed."); import traceback; traceback.print_exc()


    def create_biome_editor(self):
        """Sets up the biome editor Treeview and buttons."""
        # Needs self.biome_frame, self.scaling_manager, BiomeDialog, load_default_biomes, load_biomes_to_tree, get_biomes_from_tree
        f = self.biome_frame
        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)
        cols = ("name", "t_min", "t_max", "r_min", "r_max") # Column identifiers
        self.bt = ttk.Treeview(f, columns=cols, show="headings") # Create treeview with headings
        self.bt.heading("name", text="Name"); self.bt.column("name", width=100, stretch=tk.YES) # Set column properties

        # Update temperature column headings based on current unit system
        temp_unit = self.scaling_manager.to_display_temp(273.15)[1] # Get unit symbol (°C or °F)
        self.bt.heading("t_min", text=f"T Min ({temp_unit})"); self.bt.column("t_min", width=60, anchor='e') # 'e' for eastern (right) alignment
        self.bt.heading("t_max", text=f"T Max ({temp_unit})"); self.bt.column("t_max", width=60, anchor='e')

        self.bt.heading("r_min", text="R Min"); self.bt.column("r_min", width=60, anchor='e')
        self.bt.heading("r_max", text="R Max"); self.bt.column("r_max", width=60, anchor='e')

        self.bt.grid(row=0, column=0, sticky="nsew") # Place treeview in the grid

        # Add buttons for biome management
        bf = ttk.Frame(f)
        bf.grid(row=1, column=0, sticky="ew", pady=5)
        for i, t in enumerate(["Add", "Edit", "Remove"]):
            bf.columnconfigure(i, weight=1) # Give each button column equal weight
            # Use getattr to call the corresponding method (add_biome, edit_biome, remove_biome)
            ttk.Button(bf, text=t, command=getattr(self, f"{t.lower()}_biome")).grid(row=0, column=i, sticky="ew", padx=2)

        # Load default biomes into the treeview initially
        self.load_default_biomes() # This calls load_biomes_to_tree


    def create_formulas_panel(self):
        # ... (existing code - unchanged) ...
        f = self.formulas_frame
        f.columnconfigure(0, weight=1)

        def add_formula_editor(parent, key, title, height):
            lf = ttk.Labelframe(parent, text=title, padding=5)
            lf.grid(sticky="ew", pady=(0, 10))
            lf.columnconfigure(0, weight=1)

            text_widget = tk.Text(lf, height=height, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1, bg="#333", fg="#ddd", insertbackground="white")
            text_widget.insert("1.0", self.vars[key].get())
            text_widget.grid(sticky="ew")
            self.formula_widgets[key] = text_widget

        add_formula_editor(f, "heightmap_formula", "Heightmap Combination", 2)
        add_formula_editor(f, "temperature_base_k_formula", "Base Temperature (Kelvin)", 2)
        add_formula_editor(f, "temperature_greenhouse_k_formula", "Greenhouse Effect (Kelvin)", 2)
        add_formula_editor(f, "temperature_latitude_gradient_formula", "Latitude Gradient", 2)
        add_formula_editor(f, "temperature_lapse_rate_term_formula", "Altitude Cooling (Lapse Rate)", 2)
        add_formula_editor(f, "rainfall_orographic_effect_formula", "Orographic Rainfall (mm/yr)", 2)

        button_frame = ttk.Frame(f)
        button_frame.grid(sticky="ew", pady=10)
        button_frame.columnconfigure((0,1), weight=1)
        ttk.Button(button_frame, text="Validate Formulas", command=self.validate_formulas).grid(row=0, column=0, sticky='ew', padx=2)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_formulas).grid(row=0, column=1, sticky='ew', padx=2)


    def _get_formulas_from_ui(self):
        """Update the StringVars from the text widgets before generation."""
        for key, widget in self.formula_widgets.items():
            self.vars[key].set(widget.get("1.0", "end-1c").strip())


    def validate_formulas(self):
        """Validates the syntax of the user-defined formulas."""
        self._get_formulas_from_ui() # Ensure vars are updated from UI
        aeval = AstevalInterpreter()
        errors = []
        # Iterate through all variables, find ones ending in "_formula"
        for key, var in self.vars.items():
            if key.endswith("_formula"):
                try:
                    # Attempt to parse the formula string
                    aeval.parse(var.get())
                except Exception as e:
                    # Catch any parsing errors and add to the errors list
                    errors.append(f"Error in '{key}':\n{e}\n")

        if errors:
            # If there are errors, show a combined error message
            messagebox.showerror("Validation Failed", "".join(errors))
        else:
            # If no errors, show success message
            messagebox.showinfo("Validation Succeeded", "All formulas are syntactically correct.")


    def reset_formulas(self):
        """Resets formula text widgets and variables to default values."""
        if not messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all formulas to their default values?"):
            return
        for key, widget in self.formula_widgets.items():
            default_value = DEFAULT_FORMULAS.get(key, "")
            self.vars[key].set(default_value) # Update the variable
            widget.delete("1.0", tk.END) # Clear the text widget
            widget.insert("1.0", default_value) # Insert the default value
        self.status_var.set("Formulas have been reset to default.") # Update status bar


    def create_diagnostics_controls(self):
        """Sets up the diagnostic map viewing controls."""
        # Needs self.diagnostics_frame, self.vars["show_diagnostic_map"], self.vars["selected_diagnostic_map"], _update_diagnostic_list
        f = self.diagnostics_frame
        f.columnconfigure(0, weight=1)
        lf_diag = ttk.Labelframe(f, text="Diagnostic View", padding=5)
        lf_diag.grid(row=0, column=0, sticky="ew")
        lf_diag.columnconfigure(0, weight=1)

        # Checkbutton to enable/disable diagnostic view
        self.diag_cb = ttk.Checkbutton(lf_diag, text="Show Diagnostic Map", variable=self.vars["show_diagnostic_map"], command=self.update_display, state="disabled")
        self.diag_cb.grid(row=0, column=0, sticky="w", padx=5, pady=(5,0))
        Tooltip(self.diag_cb, "Overrides the normal display to show a single data layer from the generation process.")

        # Combobox to select the diagnostic map
        self.diag_map_cb = ttk.Combobox(lf_diag, textvariable=self.vars["selected_diagnostic_map"], state="disabled") # Start as disabled
        self.diag_map_cb.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.diag_map_cb.bind("<<ComboboxSelected>>", self.update_display) # Update display when selection changes
        Tooltip(self.diag_map_cb, "Select which intermediate data map to display.")

        # Initial update of the diagnostic list (will likely be empty at startup)
        # This call also enables/disables the controls based on data availability
        self._update_diagnostic_list() # This needs to be called after diagnostic_images and map_map_cb are created


    def create_stats_panel(self):
        """Sets up the statistics Treeview panel."""
        # Needs self.stats_frame
        f = self.stats_frame
        f.rowconfigure(0, weight=1) # Treeview row expands
        f.columnconfigure(0, weight=1) # Treeview column expands

        # Create Treeview with two columns: Statistic and Value
        self.stats_tree = ttk.Treeview(f, columns=("value",), show="tree headings") # 'tree' displays the item identifier in the first column
        self.stats_tree.heading("#0", text="Statistic") # Heading for the first column (tree column)
        self.stats_tree.column("#0", stretch=tk.YES, anchor='w') # Let the statistic column stretch
        self.stats_tree.heading("value", text="Value") # Heading for the second column (value column)
        self.stats_tree.column("value", width=120, anchor='e') # Right align the value

        self.stats_tree.grid(row=0, column=0, sticky="nsew") # Place treeview

        # Add a vertical scrollbar
        vsb = ttk.Scrollbar(f, orient="vertical", command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=vsb.set)
        vsb.grid(row=0, column=1, sticky='ns')


    def update_stats_panel(self):
        """Populates the stats panel with data from the last generation/climate simulation."""
        # Needs self.stats_tree, self.last_gen_data, self.scaling_manager, get_biomes_from_tree
        # Clear previous entries
        for i in self.stats_tree.get_children():
            self.stats_tree.delete(i)

        # Only populate if core climate data exists
        if "temperature_map" not in self.last_gen_data or "rainfall_map" not in self.last_gen_data or "land_mask" not in self.last_gen_data or "biome_indices_map" not in self.last_gen_data:
            self.stats_tree.insert("", "end", text="Run Climate Simulation to see stats.")
            return

        # Get data needed for stats
        temp_map = self.last_gen_data['temperature_map']
        rain_map = self.last_gen_data['rainfall_map']
        land_mask = self.last_gen_data['land_mask']
        indices_map = self.last_gen_data['biome_indices_map']
        albedo_map = self.last_gen_data.get('diagnostic_maps', {}).get('climate_albedo') # Get albedo if available

        # Basic Global Stats
        if albedo_map is not None:
            avg_albedo = np.mean(albedo_map)
            self.stats_tree.insert("", "end", text="Avg. Albedo", values=(f"{avg_albedo:.3f}",))

        total_pixels = land_mask.size
        land_pct = np.mean(land_mask) * 100 if total_pixels > 0 else 0
        self.stats_tree.insert("", "end", text="Land Cover", values=(f"{land_pct:.1f} %",))
        self.stats_tree.insert("", "end", text="Ocean Cover", values=(f"{100-land_pct:.1f} %",))

        if "heightmap" in self.last_gen_data:
            sea_level_norm = self.vars["sea_level"].get()
            threshold = sea_level_norm + self.scaling_manager.to_normalized(2000.0, above_sea=True)
            high_pct = np.mean(self.last_gen_data["heightmap"] > threshold) * 100
            self.stats_tree.insert("", "end", text=">2 km Elevation", values=(f"{high_pct:.2f} %",))

        avg_temp_k = np.mean(temp_map)
        disp_temp, temp_unit = self.scaling_manager.to_display_temp(avg_temp_k)
        self.stats_tree.insert("", "end", text=f"Avg. Temp ({temp_unit})", values=(f"{disp_temp:.1f}",))

        avg_rain = np.mean(rain_map)
        self.stats_tree.insert("", "end", text="Avg. Rainfall (mm/yr)", values=(f"{avg_rain:.0f}",))

        self.stats_tree.insert("", "end", iid="sep", values=("",)) # Separator line

        # Biome Distribution Stats
        self.stats_tree.insert("", "end", text="Biome Distribution", iid="biome_header") # Header item
        biomes = self.get_biomes_from_tree() # Get current biomes list

        if total_pixels > 0:
            # Flatten the indices map and count occurrences of each biome index
            flat_indices = indices_map.ravel()
            # bincount needs non-negative integers. Filter out -1 (override default) if necessary, or ensure biomes indices start from 0.
            # Assumes biome indices are 0-based and contiguous up to len(biomes)
            # Filter out the override -1 index if present
            valid_indices = flat_indices[flat_indices != -1]
            # Use minlength to ensure counts array has size at least len(biomes)
            counts = np.bincount(valid_indices, minlength=len(biomes))

            biome_dist = []
            for i, count in enumerate(counts):
                # Only include biomes that actually appear (count > 0) and have a valid index
                if count > 0 and i < len(biomes):
                    percentage = (count / total_pixels) * 100
                    biome_dist.append((biomes[i]['name'], percentage))

            # Sort biome distribution by percentage, descending
            for name, pct in sorted(biome_dist, key=lambda item: item[1], reverse=True):
                # Insert as child of the biome header
                self.stats_tree.insert("biome_header", "end", text=f"  {name}", values=(f"{pct:.2f} %",))
        else:
             self.stats_tree.insert("biome_header", "end", text="  No data available.")


    def get_biomes_from_tree(self):
        """Gets the current list of biomes from the internal biomes list."""
        # This should return the self.biomes list, which is synced with the Treeview content via load_biomes_to_tree.
        # Use this function whenever you need the current biome definitions.
        return self.biomes


    def load_biomes_to_tree(self, biomes):
        """Loads a list of biome dictionaries into the Treeview and updates the internal list."""
        # Needs self.biomes, self.bt, self.scaling_manager, update_biome_paint_list
        self.biomes = biomes # Update the internal list

        # Update temperature column headings based on current unit system
        if hasattr(self, 'bt'): # Check if treeview exists
            temp_unit = self.scaling_manager.to_display_temp(273.15)[1] # Get unit symbol (°C or °F)
            self.bt.heading("t_min", text=f"T Min ({temp_unit})")
            self.bt.heading("t_max", text=f"T Max ({temp_unit})")

            # Clear existing entries in the treeview
            for i in self.bt.get_children():
                self.bt.delete(i)

            # Insert new biome entries
            for b in self.biomes:
                # Water biomes don't have climate parameters displayed in the table
                if b.get('is_water', False):
                     self.bt.insert("", "end", values=(b['name'], '-', '-', '-', '-'), iid=b['name'])
                else:
                     # For land biomes, display climate ranges
                     self.bt.insert("", "end", values=(b['name'], b['temp_min'], b['temp_max'], b['rain_min'], b['rain_max']), iid=b['name'])

        # Update the biome list in the paint tool combobox
        self.update_biome_paint_list() # This needs self.biome_paint_cb


    def load_default_biomes(self):
        """Loads the predefined default biomes."""
        # Needs DEFAULT_BIOMES, load_biomes_to_tree
        self.load_biomes_to_tree(list(DEFAULT_BIOMES)) # Load a copy of the default list


    def add_biome(self):
        """Opens the BiomeDialog to add a new biome."""
        # Needs BiomeDialog, get_biomes_from_tree, load_biomes_to_tree
        d = BiomeDialog(self.root, self) # Open dialog, passes self (the app)
        if d.result: # If dialog returned a result (OK was clicked)
            # Add the new biome to the end of the current list and reload
            self.load_biomes_to_tree(self.get_biomes_from_tree() + [d.result])


    def edit_biome(self):
        """Opens the BiomeDialog to edit the selected biome."""
        # Needs self.bt, BiomeDialog, get_biomes_from_tree, load_biomes_to_tree
        selected_iid = self.bt.focus() # Get the id of the currently selected item
        if not selected_iid:
             messagebox.showwarning("No Selection", "Please select a biome to edit.");
             return

        # Find the biome data dictionary matching the selected item's id
        biome_data = next((b for b in self.biomes if b['name'] == selected_iid), None)
        if not biome_data:
             # This should ideally not happen if iid is the biome name
             print(f"Error: Biome data not found for selected item '{selected_iid}'.")
             return

        # Open the dialog, passing the existing data for editing
        d = BiomeDialog(self.root, self, data=biome_data)
        if d.result: # If dialog returned a result (OK was clicked)
            # Find the original biome in the list and replace it with the updated data
            for i, b in enumerate(self.biomes):
                if b['name'] == selected_iid:
                     self.biomes[i] = d.result # Replace the item in the list
                     break # Stop after finding/replacing the first match

            # Reload the entire biome list into the treeview and internal state
            self.load_biomes_to_tree(self.biomes)


    def remove_biome(self):
        """Removes the selected biome."""
        # Needs self.bt, get_biomes_from_tree, load_biomes_to_tree, messagebox
        selected_iid = self.bt.focus() # Get the id of the currently selected item
        if not selected_iid:
             messagebox.showwarning("No Selection", "Please select a biome to remove.");
             return

        # Find the biome data
        biome_data = next((b for b in self.biomes if b['name'] == selected_iid), None)
        if not biome_data:
             print(f"Error: Biome data not found for selected item '{selected_iid}'.")
             return

        # Prevent removal of essential water biomes
        if biome_data.get('is_water', False):
             messagebox.showerror("Error", "Cannot remove essential water biomes (OCEAN, INLAND_WATER).");
             return

        # Ask for confirmation before deleting
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to remove '{selected_iid}'? This cannot be undone."):
            # Create a new list excluding the biome to be removed
            self.biomes = [b for b in self.biomes if b['name'] != selected_iid]
            # Reload the updated list
            self.load_biomes_to_tree(self.biomes)
            # Note: Removing a biome means any pixels painted with its old index will likely appear as fallback biome.
            # Clearing biome overrides might be desired after removing biomes.


    def implement_rivers(self):
        """Applies the river carving effect to the heightmap."""
        # Needs self.last_gen_data, gen.calculate_river_deposition, gen.normalize_map, distance_transform_edt, PaintAction, self.history, self.scaling_manager, _update_diagnostic_list, on_history_change
        # Check if base data is available
        if "heightmap" not in self.last_gen_data or self.last_gen_data["heightmap"] is None or \
           "flow_angles" not in self.last_gen_data or self.last_gen_data["flow_angles"] is None:
             messagebox.showerror("Error", "Base world data (heightmap, flow) not found. Please generate a world first.")
             self.status_var.set("River carving failed: base data missing.")
             return

        hmap = self.last_gen_data["heightmap"]
        flow_angles = self.last_gen_data["flow_angles"]

        self.status_var.set("Recalculating river deposition and applying carving..."); self.root.update_idletasks()

        # Recalculate deposition based on current hmap and flow angles using *current* UI parameters
        # Pass a dictionary of current parameter values
        river_deposition_data = gen.calculate_river_deposition(
            hmap,
            flow_angles,
            {k: v.get() for k, v in self.vars.items()},
            self.scaling_manager,
            self.last_gen_data.get("temperature_map")
        )
        river_deposition_map = river_deposition_data['deposition_map']
        # Store the newly calculated deposition map
        self.last_gen_data['river_deposition_map'] = river_deposition_map
        # Update diagnostic maps with the raw deposition data
        self.last_gen_data.setdefault("diagnostic_maps", {}).update(river_deposition_data.get("diagnostics", {}))
        self._update_diagnostic_list() # Update the diagnostic map list in UI


        hmap_before = hmap.copy() # Save state before carving for history

        # Apply carving based on the *new* deposition map
        # Normalize deposition map to get a strength value between 0 and 1
        river_strength = gen.normalize_map(river_deposition_map)

        # Identify areas where river strength is above the user-defined threshold
        strong_river_mask = river_strength > self.vars["river_strength_threshold"].get()

        # Ensure carving only happens on land (above sea level)
        land_mask = hmap > self.vars["sea_level"].get()
        # Combine the strong river mask with the land mask
        strong_river_mask_on_land = strong_river_mask & land_mask


        if self.vars["enable_erosion"].get():
             # --- Hydraulic Erosion / Valley Carving Logic ---
             height_thresh_norm = self.vars["erosion_height_threshold"].get() # Normalized height threshold
             erodible_land_mask = hmap > height_thresh_norm # Land higher than the threshold is erodible

             # Check if there are any strong rivers on erodible land before proceeding
             if not np.any(strong_river_mask_on_land):
                  self.status_var.set("No strong rivers on erodible land to cause erosion.");
                  # Still need to regenerate visuals if only deposition map changed
                  self._regenerate_visuals_from_data(full_recalc=False)
                  self.update_display()
                  return


             # Calculate distance from any strong river point on land
             # distance_transform_edt works on a boolean mask, where True is background (distance=0)
             # So we need the inverse of the strong_river_mask_on_land
             distance_to_river = distance_transform_edt(~strong_river_mask_on_land)

             # Calculate falloff based on distance from the river, max effect at the river, falling off with distance
             # Use a arbitrary distance (e.g., 20 pixels) for falloff range
             distance_falloff_range = 20 # pixels
             # Falloff is 1 at river (distance 0), 0 at or beyond distance_falloff_range
             falloff = (1.0 - np.clip(distance_to_river / distance_falloff_range, 0, 1)) ** self.vars["erosion_sharpness"].get() # Sharpness controls the curve

             # Combine masks: Apply erosion only to erodible land near rivers (within falloff range)
             final_erosion_mask = erodible_land_mask & (distance_to_river < distance_falloff_range)

             # Determine the maximum amount of height to subtract (normalized)
             max_erosion_depth_norm = self.scaling_manager.to_normalized(
                 self.vars['max_erosion_depth_m'].get(), above_sea=True)

             # Calculate the actual erosion amount for the masked pixels
             # It scales with the maximum erosion depth and the distance falloff
             erosion_amount_masked = max_erosion_depth_norm * falloff[final_erosion_mask]

             # Apply the erosion amount to the heightmap
             hmap[final_erosion_mask] -= erosion_amount_masked


        else:
             # --- Simple Channel Carving Logic (Legacy) ---
             # Only apply carving where there are strong rivers on land
             if not np.any(strong_river_mask_on_land):
                  self.status_var.set("No strong rivers on land to carve channels.");
                  # Still need to regenerate visuals if only deposition map changed
                  self._regenerate_visuals_from_data(full_recalc=False)
                  self.update_display()
                  return

             # Base carving amount (arbitrary scaling of max erosion depth)
             carving_amount_base_norm = self.scaling_manager.to_normalized(self.vars['max_erosion_depth_m'].get() / 10.0, above_sea=True)

             # Apply carving amount scaled by the river strength *only* where strong rivers are on land
             # The river_strength map has values 0-1, so this scales the base carving amount
             hmap[strong_river_mask_on_land] -= carving_amount_base_norm * river_strength[strong_river_mask_on_land]


        # Ensure heightmap values stay within the 0.0 - 1.0 range
        np.clip(hmap, 0.0, 1.0, out=hmap)

        # Find the indices where the heightmap actually changed due to carving
        changed_indices = np.where(hmap != hmap_before)

        # Create a history action if changes occurred
        if changed_indices[0].size > 0:
            # Package the change as a PaintAction (even though it's not from painting)
            # This reuses the history mechanism for undo/redo
            action = PaintAction(hmap, changed_indices, hmap_before[changed_indices], hmap[changed_indices])
            self.history.push(action) # Add action to history stack
            self.on_history_change() # Trigger history change callback (updates visuals, menu)
            self.status_var.set("River carving complete. Use Ctrl+Z to undo.")
        else:
            # If no effective change in heightmap, but deposition map might have changed
            # Regenerate visuals to show the updated river layer image (which uses the new deposition map)
            self._regenerate_visuals_from_data(full_recalc=False)
            self.update_display() # Ensure display is updated
            self.update_edit_menu_state() # Update menu state
            self.status_var.set("No changes made to heightmap. Adjust thresholds and try again.")


    def run_generation_thread(self):
        """Starts the world generation process in a separate thread."""
        # Needs self.generate_button, self.status_var, _get_formulas_from_ui, get_biomes_from_tree, generation_worker, generation_queue, check_generation_queue
        # Disable the generate button to prevent multiple simultaneous generations
        if hasattr(self, "generate_button"): # Check if the button exists
             self.generate_button.config(state="disabled")

        self.status_var.set("Generating new world..."); # Update status bar
        self.root.update_idletasks() # Force UI update

        try:
            self._get_formulas_from_ui() # Ensure latest formulas are pulled from text widgets
            params = {k: v.get() for k, v in self.vars.items()} # Get current UI parameters
            params["biomes"] = self.get_biomes_from_tree() # Include current biomes in params

        except (tk.TclError, ValueError) as e:
            # Handle invalid input from UI controls
            messagebox.showerror("Invalid Input", f"Invalid setting: {e}")
            if hasattr(self, "generate_button"): self.generate_button.config(state="normal") # Re-enable button
            return # Stop generation if input is invalid

        # Start the generation worker thread, passing parameters and the queue
        # The worker will put its result (or an exception) into the queue
        threading.Thread(target=self.generation_worker, args=(params, self.generation_queue), daemon=True).start() # daemon=True lets thread exit with app

        # Start checking the queue periodically for results
        self.root.after(100, self.check_generation_queue)


    def generation_worker(self, params, q):
        """Worker thread function for the initial world data generation."""
        # Needs gen.generate_world_data
        # Puts result or exception into the queue q
        try:
            # generate_world_data now calculates base maps, flow angles, and river deposition map
            world_data = gen.generate_world_data(params, self.scaling_manager)
            q.put(world_data) # Put the successful result into the queue
        except Exception as e:
            # Catch any exceptions during generation and put them into the queue
            q.put(e)


    def check_generation_queue(self):
        """Checks the queue for results from the generation worker thread."""
        # Needs self.generation_queue, self.last_gen_data, run_climate_thread, _regenerate_visuals_from_data, update_display, current_filepath, update_edit_menu_state, status_var
        # Needs self.generate_button (optional)
        try:
            result = self.generation_queue.get_nowait() # Try to get result without blocking

            if hasattr(self, "generate_button"): # Re-enable button regardless of success/failure
                 self.generate_button.config(state="normal")

            if isinstance(result, Exception):
                # Handle errors received from the worker thread
                self.status_var.set("Error during generation.");
                messagebox.showerror("Error", f"An error occurred: {result}");
                # Print traceback to console for debugging
                import traceback; traceback.print_exc()
                # Clear any potentially incomplete last_gen_data? Or leave it for inspection?
                # Let's leave it, but maybe remove keys that are None or failed.
                # Or just rely on subsequent checks before using data.

            else:
                # Generation was successful
                self.last_gen_data = result # Store all the generated data

                # --- Start the climate simulation process (now separate) ---
                # The climate simulation needs heightmap, flow angles, and river deposition data,
                # which are now included in self.last_gen_data from generate_world_data.
                # It also needs temp_map, rain_map for classification, which it generates.
                self.run_climate_thread() # Call the function that runs climate simulation

                # After the climate thread finishes (handled by its own status updates),
                # the display will be updated.

                # Reset project state for a new world
                self.current_filepath = None # Mark project as unsaved
                self.root.title("Advanced World Editor - New World*")
                self.history.undo_stack.clear(); self.history.redo_stack.clear() # Clear history
                self.update_edit_menu_state() # Update menu states (undo/redo disabled)

                self.status_var.set("Generation complete. Running climate simulation...") # Initial status after generation

        except queue.Empty:
            # Queue is empty, worker is still running
            # Check the queue again after a short delay
            self.root.after(200, self.check_generation_queue)


    def run_climate_thread(self):
        """Starts the climate simulation process."""
        # Needs apply_climate_simulation, _regenerate_visuals_from_data, update_display, update_stats_panel, status_var
        # This function is now called after initial generation completes in check_generation_queue.
        # It primarily wraps the call to apply_climate_simulation.

        # apply_climate_simulation performs the core climate calculations (temp, rain, classification).
        # It updates self.last_gen_data with 'temperature_map', 'rainfall_map', 'biome_image', etc.
        # Error handling is inside apply_climate_simulation.
        self.apply_climate_simulation()

        # After climate simulation is attempted (whether successful or not),
        # regenerate all visual layers based on the current state of self.last_gen_data.
        self.status_var.set("Climate simulation complete. Generating visuals..."); # Update status bar before visuals
        self.root.update_idletasks()
        self._regenerate_visuals_from_data(full_recalc=True) # Regenerate all visual layers

        # Update the display with the newly generated visuals
        self.update_display(is_new_generation=True)

        # Update the stats panel
        self.update_stats_panel()

        self.status_var.set("Climate and biome simulation complete.") # Final status update

        # Note: If apply_climate_simulation fails, it shows an error box and sets status.
        # This function continues, attempts to regenerate visuals (might fail partially if data is None),
        # updates display (might show partial data), and updates stats (might show incomplete stats).
        # This seems like acceptable behavior - show whatever data was successfully generated.


    def _regenerate_visuals_from_data(self, full_recalc=False):
        """Regenerates all image layers based on the current state of the data maps."""
        # Needs self.last_gen_data, gen functions (create_heatmap_image, create_river_image, create_flow_map_image, apply_altitude_tint), self.vars, get_biomes_from_tree, self.scaling_manager
        # Updates self.last_gen_data with image objects (_image, _layer, _tinted)
        # Calls update_display at the end
        if "heightmap" not in self.last_gen_data or self.last_gen_data["heightmap"] is None:
            # No heightmap to base visuals on
            self.status_var.set("No heightmap data to generate visuals.")
            self.map_canvas.set_image(Image.new("RGBA", (512, 512), (0, 0, 0, 0)))  # blank image
            self.map_canvas.fit_to_screen()
            self.scroll_x.set(self.map_canvas.get_x_offset_ratio())
            self._show_initial_message()  # Show initial message
            return  # Cannot regenerate visuals without heightmap

        self.status_var.set("Regenerating visuals..."); # Update status bar
        self.root.update_idletasks() # Force UI update

        hmap = self.last_gen_data["heightmap"]
        h, w = hmap.shape # Get dimensions from heightmap
        params = {k: v.get() for k, v in self.vars.items()} # Get current UI parameters
        biomes = self.get_biomes_from_tree() # Get current biomes list


        # 1. Create Heightmap Image (Always possible if hmap exists)
        try:
             self.last_gen_data["heightmap_image"] = gen.create_heatmap_image(hmap)
        except Exception as e:
             print(f"Error creating heightmap image: {e}")
             self.last_gen_data["heightmap_image"] = Image.new("RGBA", (w, h), (255, 0, 0, 128)) # Red overlay to indicate error


        # 2. Create River Layer Image (Needs deposition map and land mask)
        # Deposition map comes from generate_world_data or calculate_river_deposition
        # Land mask comes from generate_classification_maps
        # If land_mask is missing, try to generate a simple one from heightmap
        river_deposition_map = self.last_gen_data.get("river_deposition_map")
        if river_deposition_map is not None:
             land_mask = self.last_gen_data.get("land_mask")
             if land_mask is None:
                  # Fallback: generate land mask from heightmap if climate sim hasn't run or failed
                  sea_level_norm = params.get("sea_level", SEA_LEVEL)
                  land_mask = hmap > sea_level_norm
                  print("Warning: Land mask missing for river visual, generating from heightmap.")

             try:
                  self.last_gen_data["river_layer"] = gen.create_river_image(river_deposition_map, land_mask)
             except Exception as e:
                  print(f"Error creating river image: {e}")
                  self.last_gen_data["river_layer"] = Image.new("RGBA", (w, h), (0, 0, 255, 128)) # Blue overlay for error


        # 3. Create Tectonic Flow Image (Needs plate points/velocities)
        # These come from generate_world_data
        plate_points = self.last_gen_data.get("plate_points")
        plate_velocities = self.last_gen_data.get("plate_velocities")
        if plate_points is not None and plate_velocities is not None:
             try:
                  self.last_gen_data["flow_map_image"] = gen.create_flow_map_image(plate_points, plate_velocities, w, h)
             except Exception as e:
                  print(f"Error creating tectonic flow image: {e}")
                  self.last_gen_data["flow_map_image"] = Image.new("RGBA", (w, h), (255, 255, 0, 128)) # Yellow overlay for error


        # --- Climate-Dependent Visuals ---
        # These require temperature, rainfall, and biome classification data
        temp_map = self.last_gen_data.get("temperature_map")
        rain_map = self.last_gen_data.get("rainfall_map")
        biome_image = self.last_gen_data.get("biome_image") # Created by generate_classification_maps


        if temp_map is not None and rain_map is not None and biome_image is not None:
             # Climate simulation has successfully run

             # 4. Create Temperature Image
             try:
                  self.last_gen_data["temperature_image"] = gen.create_temperature_image(temp_map)
             except Exception as e:
                  print(f"Error creating temperature image: {e}")
                  self.last_gen_data["temperature_image"] = Image.new("RGBA", (w, h), (255, 0, 255, 128)) # Magenta overlay

             # 5. Create Rainfall Image
             try:
                  self.last_gen_data["rainfall_image"] = gen.create_rainfall_image(rain_map)
             except Exception as e:
                  print(f"Error creating rainfall image: {e}")
                  self.last_gen_data["rainfall_image"] = Image.new("RGBA", (w, h), (0, 255, 255, 128)) # Cyan overlay


             # 6. Apply Altitude Tint (Needs biome_image, hmap, sea_level, scaling_manager, temp_map, rain_map)
             # Only apply if enabled in UI
             if self.vars["enable_altitude_tint"].get():
                  self.status_var.set("Applying altitude tint..."); # Update status bar
                  self.root.update_idletasks()
                  try:
                       sea_level_norm = params.get("sea_level", SEA_LEVEL)
                       # Pass a copy of biome_image to avoid modifying the original
                       tinted_image = gen.apply_altitude_tint(
                            biome_image.copy(), # Use the base biome image
                            hmap,
                            sea_level_norm,
                            self.scaling_manager,
                            temp_map, # Pass temperature map
                            rain_map   # Pass rainfall map
                       )
                       self.last_gen_data["biome_image_tinted"] = tinted_image
                       self.status_var.set("Altitude tint applied.")
                  except Exception as e:
                       print(f"Error applying altitude tint: {e}")
                       self.status_var.set("Error applying altitude tint.")
                       # If tint fails, fallback to the non-tinted biome image for display
                       self.last_gen_data["biome_image_tinted"] = biome_image # Use the base biome image


             else:
                  # If altitude tint is disabled in UI, just alias biome_image_tinted to biome_image
                  # This simplifies display logic below.
                  self.last_gen_data["biome_image_tinted"] = biome_image # Alias

        else:
             # Climate simulation data is not available, ensure climate-dependent images are None
             self.last_gen_data["temperature_image"] = None
             self.last_gen_data["rainfall_image"] = None
             self.last_gen_data["biome_image_tinted"] = self.last_gen_data.get("biome_image") # Use base biome image if it exists


        # Update diagnostic list as new maps might have been added
        self._update_diagnostic_list()

        # Finally, update the canvas display with the new layers
        self.update_display() # This function composites the layers


        self.status_var.set("Visuals ready.") # Final status update after regeneration and display update


    def update_display(self, event=None, is_new_generation=False):
        """Composites the image layers based on visibility settings and updates the canvas."""
        # Needs self.last_gen_data (image objects), self.vars (layer visibility), self.map_canvas, _update_diagnostic_list
        # Needs gen functions (create_angle_map_image, create_temperature_image, etc.) if showing diagnostic
        if "heightmap" not in self.last_gen_data or self.last_gen_data["heightmap"] is None:
             # No heightmap data, show the initial message
             self._show_initial_message()
             return # Exit display update

        # --- Diagnostic Map Display ---
        # If diagnostic view is enabled and a diagnostic map image is available, display it directly
        if self.vars["show_diagnostic_map"].get():
            selected_map_name = self.vars["selected_diagnostic_map"].get()
            if (
                selected_map_name in self.diagnostic_images
                and self.diagnostic_images[selected_map_name] is not None
            ):
                diagnostic_image = self.diagnostic_images[selected_map_name]
                self.map_canvas.set_image(diagnostic_image)
                if is_new_generation:
                    self.map_canvas.fit_to_screen()
                    self.scroll_x.set(self.map_canvas.get_x_offset_ratio())
                else:
                    self.map_canvas.redraw()

                # Update status bar with specific diagnostic value under cursor
                diag_data_map = self.last_gen_data.get("diagnostic_maps", {}).get(
                    selected_map_name
                )
                if (
                    diag_data_map is not None
                    and diag_data_map.shape == self.last_gen_data["heightmap"].shape
                ):
                    # Get mouse position relative to canvas window
                    mouse_x = self.root.winfo_pointerx() - self.map_canvas.winfo_rootx()
                    mouse_y = self.root.winfo_pointery() - self.map_canvas.winfo_rooty()
                    # Convert canvas coordinates to world coordinates
                    mx, my = self.map_canvas.canvas_to_world(mouse_x, mouse_y)
                    h, w = self.last_gen_data["heightmap"].shape
                    if 0 <= mx < w and 0 <= my < h:
                        ix, iy = int(mx), int(my)
                        value = diag_data_map[iy, ix]
                        value_str = (
                            f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
                        )
                        self.status_var.set(
                            f"Diagnostic: {selected_map_name} | X: {ix}, Y: {iy} | Value: {value_str} | Zoom: {self.map_canvas.zoom_level:.2f}x"
                        )
                    else:
                        self.status_var.set(
                            f"Diagnostic: {selected_map_name} | Zoom: {self.map_canvas.zoom_level:.2f}x"
                        )
                else:
                    self.status_var.set(
                        f"Diagnostic: {selected_map_name} | (Data N/A) | Zoom: {self.map_canvas.zoom_level:.2f}x"
                    )

                return  # Exit the function after showing diagnostic


        # --- Layer Compositing Display ---
        self.status_var.set("Compositing layers...") # Update status bar


        # Determine the base image layer based on visibility and availability
        # Prioritize tinted biome > biome > heightmap > transparent black
        base_image = None
        hmap_shape = self.last_gen_data["heightmap"].T.shape # (W, H) for PIL Image.new


        use_biome_layer = self.layer_vars["Biomes"].get()
        use_heightmap_layer = self.layer_vars["Heightmap"].get()
        use_tinted_biome = self.layer_vars["Altitude Tint"].get() and "biome_image_tinted" in self.last_gen_data and self.last_gen_data["biome_image_tinted"] is not None
        biome_image_exists = "biome_image" in self.last_gen_data and self.last_gen_data["biome_image"] is not None
        heightmap_image_exists = "heightmap_image" in self.last_gen_data and self.last_gen_data["heightmap_image"] is not None


        if use_biome_layer and use_tinted_biome:
            # Use the tinted biome image as the base if both Biomes and Altitude Tint are enabled
            base_image = self.last_gen_data["biome_image_tinted"].convert("RGBA")
        elif use_biome_layer and biome_image_exists:
            # Otherwise, if Biomes is enabled, use the regular biome image
            base_image = self.last_gen_data["biome_image"].convert("RGBA")
        elif use_heightmap_layer and heightmap_image_exists:
            # If Biomes is off, but Heightmap is enabled, use the heightmap image as base
            base_image = self.last_gen_data["heightmap_image"].copy() # Copy to avoid modifying original
            if base_image.mode != 'RGBA': base_image = base_image.convert('RGBA') # Ensure RGBA for blending/pasting
        else:
            # If no base layer is selected or available, start with a transparent black image
            base_image = Image.new("RGBA", hmap_shape, (0, 0, 0, 0)) # Transparent black


        # --- Blend/Paste Optional Layers ---
        # Ensure base image is RGBA before blending/pasting other layers
        if base_image.mode != 'RGBA':
             base_image = base_image.convert('RGBA')


        # Blend Heightmap image on top if Heightmap layer is on AND it wasn't the base image
        # We blend it with a low alpha (e.g., 0.5) to show underlying layers
        if use_heightmap_layer and heightmap_image_exists and base_image != self.last_gen_data["heightmap_image"]:
             heightmap_blend_image = self.last_gen_data["heightmap_image"].copy() # Copy for blending
             if heightmap_blend_image.mode != 'RGBA': heightmap_blend_image = heightmap_blend_image.convert('RGBA')
             base_image = Image.blend(base_image, heightmap_blend_image, alpha=0.5)


        # Blend Temperature image if enabled
        if "temperature_image" in self.last_gen_data and self.last_gen_data["temperature_image"] is not None and self.layer_vars["Temperature"].get():
            temp_image = self.last_gen_data["temperature_image"].copy() # Copy for blending
            if temp_image.mode != 'RGBA': temp_image = temp_image.convert('RGBA')
            base_image = Image.blend(base_image, temp_image, alpha=0.5)


        # Blend Rainfall image if enabled
        if "rainfall_image" in self.last_gen_data and self.last_gen_data["rainfall_image"] is not None and self.layer_vars["Rainfall"].get():
            rain_image = self.last_gen_data["rainfall_image"].copy() # Copy for blending
            if rain_image.mode != 'RGBA': rain_image = rain_image.convert('RGBA')
            base_image = Image.blend(base_image, rain_image, alpha=0.5)


        # Paste River layer if enabled (uses its alpha channel for transparency)
        if "river_layer" in self.last_gen_data and self.last_gen_data["river_layer"] is not None and self.layer_vars["Rivers"].get():
            river_image = self.last_gen_data["river_layer"].copy() # Copy for pasting
            if river_image.mode == 'RGBA':
                 base_image.paste(river_image, (0, 0), river_image) # Use image itself as mask for RGBA
            else: # Fallback if somehow not RGBA
                 base_image.paste(river_image, (0, 0))


        # Paste Tectonic Flow layer if enabled (uses its alpha channel)
        if "flow_map_image" in self.last_gen_data and self.last_gen_data["flow_map_image"] is not None and self.layer_vars["Tectonic Plates"].get():
            flow_image = self.last_gen_data["flow_map_image"].copy() # Copy for pasting
            if flow_image.mode == 'RGBA':
                 base_image.paste(flow_image, (0, 0), flow_image) # Use image itself as mask for RGBA
            else: # Fallback
                 base_image.paste(flow_image, (0, 0))


        # Set the final composite image to the canvas
        self.map_canvas.set_image(base_image)

        # Redraw the canvas or fit to screen for a new generation
        if is_new_generation:
             self.map_canvas.fit_to_screen()
             self.scroll_x.set(self.map_canvas.get_x_offset_ratio())
        else:
             self.map_canvas.redraw() # Just redraw with current zoom/pan


        # Update status bar (mouse move handler updates location details)
        # Set a general status message here after compositing finishes
        # Check if currently painting to avoid overwriting detailed paint status
        if not self.is_painting:
             self.status_var.set(f"Display updated. Zoom: {self.map_canvas.zoom_level:.2f}x")


    def save_project(self):
        """Saves the current project data to the last used file."""
        # Needs self.current_filepath, save_as_project, last_gen_data, _perform_save, messagebox
        if not self.current_filepath:
             # If no file path is set, call save_as_project
             self.save_as_project()
        elif self.last_gen_data and "heightmap" in self.last_gen_data and self.last_gen_data["heightmap"] is not None:
             # If a file path is set and there's data to save, perform the save
             self._perform_save(self.current_filepath)
        else:
             # No data to save
             messagebox.showerror("Error", "No map data available to save.");
             self.status_var.set("Save failed: no map data.")


    def save_as_project(self):
        """Prompts the user for a file location and saves the current project data."""
        # Needs last_gen_data, _perform_save, filedialog, messagebox
        if not self.last_gen_data or "heightmap" not in self.last_gen_data or self.last_gen_data["heightmap"] is None:
             messagebox.showerror("Error", "No map data available to save.");
             self.status_var.set("Save failed: no map data.")
             return

        # Open a file dialog to choose save location
        fp = filedialog.asksaveasfilename(
             defaultextension=".npz", # Default extension
             filetypes=[("World Project", "*.npz"), ("NumPy compressed", "*.npz"), ("All files", "*.*")], # File type options
             title="Save World Project As..." # Dialog title
        )

        if fp: # If user selected a file path (didn't cancel)
             self._perform_save(fp) # Perform the actual save operation


    def _perform_save(self, fp):
        """Performs the actual saving of project data to a file."""
        self.status_var.set(f"Saving to {os.path.basename(fp)}...");
        self.root.update_idletasks()

        try:
            self._get_formulas_from_ui()
            params_to_save = {k: v.get() for k, v in self.vars.items() if k not in ["show_diagnostic_map", "selected_diagnostic_map"]}
            biomes_to_save = self.get_biomes_from_tree()

            if "heightmap" not in self.last_gen_data or self.last_gen_data["heightmap"] is None:
                raise ValueError("No heightmap data available to save.")

            hmap = self.last_gen_data["heightmap"]
            data_to_save = {
                "heightmap": hmap.copy(),
                "biome_override_map": self.last_gen_data.get("biome_override_map", np.full_like(hmap, -1, dtype=np.int8)).copy(),
                "params": json.dumps(params_to_save),
                "biomes": json.dumps(biomes_to_save)
            }

            optional_keys = [
                "flow_angles", "river_deposition_map", "temperature_map",
                "rainfall_map", "biome_indices_map", "land_mask"
            ]
            for key in optional_keys:
                if key in self.last_gen_data and self.last_gen_data[key] is not None:
                    data_to_save[key] = self.last_gen_data[key].copy()

            np.savez_compressed(fp, **data_to_save)

            self.current_filepath = fp
            self.root.title(f"Advanced World Editor - {os.path.basename(fp)}")
            self.status_var.set("Project saved.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save file:\n\n{e}")
            self.status_var.set("Save failed.")
            import traceback; traceback.print_exc()

    def open_project(self):
        """Prompts the user for a file and loads the project data."""
        fp = filedialog.askopenfilename(
             filetypes=[("World Project", "*.npz"), ("All files", "*.*")],
             title="Open World Project"
        )
        if not fp:
             return

        self.status_var.set(f"Opening {os.path.basename(fp)}...");
        self.root.update_idletasks()

        try:
            with np.load(fp, allow_pickle=True) as data:
                if "params" in data:
                    params = json.loads(data["params"].item())
                    for k, v in params.items():
                        if k in self.vars:
                            try:
                                self.vars[k].set(v)
                            except tk.TclError as e:
                                print(f"Warning: Could not set variable '{k}': {e}")

                for key, widget in self.formula_widgets.items():
                    widget.delete("1.0", tk.END)
                    widget.insert("1.0", self.vars.get(key, "").get())

                self.last_gen_data = {}
                for key in data.files:
                    if key not in ["params", "biomes"]:
                        self.last_gen_data[key] = data[key]

                if "biomes" in data:
                    biomes = json.loads(data["biomes"].item())
                    self.load_biomes_to_tree(biomes)
                else:
                    self.load_default_biomes()

                self.vars["show_diagnostic_map"].set(False)

            self._on_unit_or_scale_change()
            self._regenerate_visuals_from_data(full_recalc=True)
            self.update_display(is_new_generation=True)
            self.current_filepath = fp
            self.root.title(f"Advanced World Editor - {os.path.basename(fp)}")
            self.status_var.set("Project loaded.")
            self.history.undo_stack.clear()
            self.history.redo_stack.clear()
            self.update_edit_menu_state()
            if "temperature_map" in self.last_gen_data:
                self.update_stats_panel()

        except Exception as e:
            messagebox.showerror("Load Error", f"An error occurred while loading file:\n\n{e}")
            self.status_var.set("Load failed.")
            import traceback; traceback.print_exc()



if __name__ == "__main__":
    # --- Main application entry point ---
    # Create the themed Tkinter root window
    root = ThemedTk(theme="equilux") # Using a themed window

    # Run the application
    try:
        app = MapGeneratorApp(root) # Create an instance of the main application class
        root.mainloop() # Start the Tkinter event loop
    except Exception as e:
        # Catch any unexpected errors that escape the application logic
        messagebox.showerror("Fatal Error", f"An unexpected error occurred and the application must close:\n\n{e}")
        # Print traceback for debugging the fatal error
        import traceback; traceback.print_exc()