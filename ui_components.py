# advanced-world-editor/ui_components.py

import tkinter as tk
from tkinter import ttk, colorchooser, messagebox
from PIL import Image, ImageTk

# --- A ROBUST, ZOOMABLE CANVAS WIDGET ---
class ZoomableCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.zoom_level = 1.0
        self.x_offset = 0
        self.y_offset = 0
        self.pil_image = None
        self.tk_image = None
        self.image_id = None

    def set_image(self, pil_image):
        self.pil_image = pil_image

    def fit_to_screen(self):
        if not self.pil_image: return
        self.delete("all")
        self.image_id = None
        canvas_w, canvas_h = self.winfo_width(), self.winfo_height()
        img_w, img_h = self.pil_image.size
        if canvas_w < 2 or canvas_h < 2 or img_w < 1 or img_h < 1: return
        ratio = min(canvas_w / img_w, canvas_h / img_h) * 0.95
        self.zoom_level = max(0.1, min(10.0, ratio))
        self.x_offset = max(0.0, (img_w - canvas_w / self.zoom_level) / 2)
        self.y_offset = max(0.0, (img_h - canvas_h / self.zoom_level) / 2)
        self.redraw()

    def redraw(self):
        if not self.pil_image: return
        canvas_w, canvas_h = self.winfo_width(), self.winfo_height()
        if canvas_w < 2 or canvas_h < 2: return
        img_w, img_h = self.pil_image.size
        # Clamp offsets so the crop box stays within the image bounds
        max_x_off = max(0.0, img_w - canvas_w / self.zoom_level)
        max_y_off = max(0.0, img_h - canvas_h / self.zoom_level)
        self.x_offset = min(max(self.x_offset, 0.0), max_x_off)
        self.y_offset = min(max(self.y_offset, 0.0), max_y_off)

        box_x1, box_y1 = self.x_offset, self.y_offset
        box_x2 = box_x1 + canvas_w / self.zoom_level
        box_y2 = box_y1 + canvas_h / self.zoom_level
        cropped_image = self.pil_image.crop((box_x1, box_y1, box_x2, box_y2))
        # Use high-quality resampling to avoid aliasing artifacts that look like
        # a black/white checkerboard when displaying large images at small
        # scales.
        resized_image = cropped_image.resize(
            (canvas_w, canvas_h), Image.Resampling.LANCZOS
        )
        self.tk_image = ImageTk.PhotoImage(resized_image)
        if self.image_id:
            self.itemconfig(self.image_id, image=self.tk_image)
        else:
            self.image_id = self.create_image(0, 0, anchor="nw", image=self.tk_image)

    def pan(self, dx, dy):
        if not self.pil_image:
            return
        img_w, img_h = self.pil_image.size
        self.x_offset -= dx / self.zoom_level
        self.y_offset -= dy / self.zoom_level
        canvas_w, canvas_h = self.winfo_width(), self.winfo_height()
        max_x_off = max(0.0, img_w - canvas_w / self.zoom_level)
        max_y_off = max(0.0, img_h - canvas_h / self.zoom_level)
        self.x_offset = min(max(self.x_offset, 0.0), max_x_off)
        self.y_offset = min(max(self.y_offset, 0.0), max_y_off)
        self.redraw()

    def zoom(self, event):
        if not self.pil_image: return
        factor = 1.1 if event.delta > 0 else 0.9
        world_x_before, world_y_before = self.canvas_to_world(event.x, event.y)
        self.zoom_level *= factor
        self.zoom_level = max(0.1, min(10.0, self.zoom_level))
        world_x_after, world_y_after = self.canvas_to_world(event.x, event.y)
        self.x_offset += world_x_before - world_x_after
        self.y_offset += world_y_before - world_y_after
        self.redraw()

    def canvas_to_world(self, canvas_x, canvas_y):
        return (self.x_offset + canvas_x / self.zoom_level, self.y_offset + canvas_y / self.zoom_level)

# --- TOOLTIP CLASS ---
class Tooltip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tooltip_window = widget, text, None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        if self.tooltip_window or not self.text: return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        ttk.Label(tw, text=self.text, justify="left", background="#333333", foreground="white", relief="solid", borderwidth=1, wraplength=250, padding=5).pack(ipadx=1)

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

# --- BIOME DIALOG ---
class BiomeDialog(tk.Toplevel):
    def __init__(self, parent, app, data=None):
        super().__init__(parent)
        self.transient(parent)
        self.title("Edit Biome" if data else "Add Biome")
        self.result = None
        self.app = app
        self.sm = app.scaling_manager
        
        # Determine if it's a water biome (which has no climate params)
        self.is_water_biome = data.get('is_water', False) if data else False
        
        # --- Variables ---
        self.name_var = tk.StringVar(value=data.get("name", ""))
        self.color_var = tk.StringVar(value=data.get("color", "#ffffff"))
        self.temp_min_var = tk.DoubleVar(value=data.get("temp_min", 0))
        self.temp_max_var = tk.DoubleVar(value=data.get("temp_max", 20))
        self.rain_min_var = tk.DoubleVar(value=data.get("rain_min", 500))
        self.rain_max_var = tk.DoubleVar(value=data.get("rain_max", 1500))

        # --- Widgets ---
        frame = ttk.Frame(self, padding=10)
        frame.pack(expand=True, fill=tk.BOTH)

        # Name and Color
        ttk.Label(frame, text="Name:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(frame, textvariable=self.name_var, state="readonly" if self.is_water_biome else "normal").grid(row=0, column=1, columnspan=2, sticky="ew", pady=2)
        
        ttk.Label(frame, text="Color:").grid(row=1, column=0, sticky="w", pady=2)
        self.color_label = ttk.Label(frame, text=self.color_var.get(), background=self.color_var.get(), width=10, relief="solid")
        self.color_label.grid(row=1, column=1, sticky="w", pady=2)
        ttk.Button(frame, text="Choose...", command=self.choose_color).grid(row=1, column=2, sticky="e", pady=2)

        # Separator and Climate controls (for land biomes only)
        if not self.is_water_biome:
            ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=2, column=0, columnspan=3, sticky="ew", pady=10)
            
            temp_unit = self.sm.to_display_temp(273.15)[1]
            ttk.Label(frame, text=f"Temperature ({temp_unit}):").grid(row=3, column=0, sticky="w", pady=2)
            ttk.Entry(frame, textvariable=self.temp_min_var, width=8).grid(row=3, column=1, sticky="e", pady=2)
            ttk.Entry(frame, textvariable=self.temp_max_var, width=8).grid(row=3, column=2, sticky="w", padx=5, pady=2)
            
            ttk.Label(frame, text="Rainfall (mm/yr):").grid(row=4, column=0, sticky="w", pady=2)
            ttk.Entry(frame, textvariable=self.rain_min_var, width=8).grid(row=4, column=1, sticky="e", pady=2)
            ttk.Entry(frame, textvariable=self.rain_max_var, width=8).grid(row=4, column=2, sticky="w", padx=5, pady=2)

        # OK/Cancel buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="OK", command=self.on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)

        self.grab_set()
        self.wait_window(self)

    def choose_color(self):
        color_code = colorchooser.askcolor(title="Choose Color", initialcolor=self.color_var.get())
        if color_code and color_code[1]:
            self.color_var.set(color_code[1])
            self.color_label.config(text=color_code[1], background=color_code[1])

    def on_ok(self):
        name = self.name_var.get()
        if not name:
            messagebox.showerror("Invalid Input", "Biome name cannot be empty.", parent=self)
            return

        self.result = {"name": name, "color": self.color_var.get()}
        if self.is_water_biome:
            self.result["is_water"] = True
        else:
            try:
                self.result.update({
                    "temp_min": self.temp_min_var.get(),
                    "temp_max": self.temp_max_var.get(),
                    "rain_min": self.rain_min_var.get(),
                    "rain_max": self.rain_max_var.get()
                })
            except tk.TclError:
                messagebox.showerror("Invalid Input", "Climate values must be valid numbers.", parent=self)
                return
        
        self.destroy()