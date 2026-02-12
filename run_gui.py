import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from pathlib import Path
import threading
import sys
import os
from PIL import Image, ImageTk
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    from texture_generator import MinecraftTextureAI
except ImportError:
    messagebox.showerror(
        "Import Error",
        "Could not import texture_generator.py\n\n"
        "Please run INSTALL.py first to install dependencies!"
    )
    sys.exit(1)

class TextureAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Minecraft 16x16 Texture AI Generator")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)

        self.ai = MinecraftTextureAI(str(Path(__file__).parent))
        self.current_texture_type = "blocks"
        self.current_category = None
        self.last_generated_path = None

        self.setup_style()

        self.create_menu()
        self.create_main_layout()

        self.refresh_categories()
    
    def setup_style(self):
        
        style = ttk.Style()
        style.theme_use('clam')

        bg_color = "#2b2b2b"
        fg_color = "#ffffff"
        accent_color = "#4a9eff"
        
        self.root.configure(bg=bg_color)
        
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color, font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=10)
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"))
        
        style.map("TButton",
                  background=[('active', accent_color)],
                  foreground=[('active', 'white')])
    
    def create_menu(self):
        
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Training Folder", command=self.open_training_folder)
        file_menu.add_command(label="Open Output Folder", command=self.open_output_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Quick Start Guide", command=self.show_quickstart)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_main_layout(self):

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        title = ttk.Label(main_frame, text="🎮 Minecraft Texture AI", style="Title.TLabel")
        title.grid(row=0, column=0, pady=10)

        content = ttk.Frame(main_frame)
        content.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.columnconfigure(2, weight=2)

        self.create_training_panel(content)

        self.create_generation_panel(content)

        self.create_preview_panel(content)
    
    def create_training_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="📚 Training", padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        parent.rowconfigure(0, weight=1)
        
        canvas = tk.Canvas(frame, bg="#2b2b2b", highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        ttk.Label(scrollable_frame, text="Texture Type:").pack(anchor=tk.W, pady=5)
        
        self.train_type_var = tk.StringVar(value="blocks")
        type_frame = ttk.Frame(scrollable_frame)
        type_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(type_frame, text="Blocks", variable=self.train_type_var, 
                       value="blocks").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(type_frame, text="Items", variable=self.train_type_var,
                       value="items").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(type_frame, text="Both", variable=self.train_type_var,
                       value="all").pack(side=tk.LEFT, padx=5)
        
        info_frame = ttk.Frame(scrollable_frame)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.training_info = tk.Text(info_frame, height=10, width=30, 
                                     bg="#1e1e1e", fg="#ffffff",
                                     font=("Consolas", 9))
        self.training_info.pack(fill=tk.BOTH, expand=True)
        
        epoch_frame = ttk.Frame(scrollable_frame)
        epoch_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(epoch_frame, text="Shading Epochs:").pack(side=tk.LEFT)
        self.shading_epochs = tk.StringVar(value="500")
        ttk.Entry(epoch_frame, textvariable=self.shading_epochs, width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(epoch_frame, text="Color Epochs:").pack(side=tk.LEFT, padx=(10, 0))
        self.color_epochs = tk.StringVar(value="300")
        ttk.Entry(epoch_frame, textvariable=self.color_epochs, width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(scrollable_frame, text="🚀 Start Training", 
                  command=self.start_training, style="Accent.TButton").pack(fill=tk.X, pady=10)
        
        ttk.Button(scrollable_frame, text="📁 Open Training Folder",
                  command=self.open_training_folder).pack(fill=tk.X, pady=5)
        
        self.update_training_info()
    
    def create_generation_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="✨ Generate Texture", padding="10")
        frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        canvas = tk.Canvas(frame, bg="#2b2b2b", highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        ttk.Label(scrollable_frame, text="Type:").pack(anchor=tk.W, pady=5)
        
        type_buttons = ttk.Frame(scrollable_frame)
        type_buttons.pack(fill=tk.X, pady=5)
        
        ttk.Button(type_buttons, text="Blocks", 
                  command=lambda: self.set_texture_type("blocks")).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(type_buttons, text="Items",
                  command=lambda: self.set_texture_type("items")).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        ttk.Label(scrollable_frame, text="Category:").pack(anchor=tk.W, pady=(10, 5))
        
        self.category_var = tk.StringVar()
        self.category_dropdown = ttk.Combobox(scrollable_frame, textvariable=self.category_var, 
                                             state="readonly", width=25)
        self.category_dropdown.pack(fill=tk.X, pady=5)
        
        ttk.Button(scrollable_frame, text="🔄 Refresh Categories",
                  command=self.refresh_categories).pack(fill=tk.X, pady=5)
        
        ttk.Label(scrollable_frame, text="Output Name:").pack(anchor=tk.W, pady=(10, 5))
        self.output_name_var = tk.StringVar(value="my_texture")
        ttk.Entry(scrollable_frame, textvariable=self.output_name_var).pack(fill=tk.X, pady=5)
        
        ttk.Label(scrollable_frame, text="Seed (optional):").pack(anchor=tk.W, pady=(10, 5))
        self.seed_var = tk.StringVar()
        seed_frame = ttk.Frame(scrollable_frame)
        seed_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(seed_frame, textvariable=self.seed_var, width=15).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(seed_frame, text="Random", 
                  command=lambda: self.seed_var.set("")).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(scrollable_frame, text="Smoothness (reduces grain):").pack(anchor=tk.W, pady=(10, 5))
        self.smoothness_var = tk.DoubleVar(value=0.3)
        smoothness_frame = ttk.Frame(scrollable_frame)
        smoothness_frame.pack(fill=tk.X, pady=5)
        
        smoothness_slider = tk.Scale(smoothness_frame, from_=0.0, to=1.0, resolution=0.1,
                                     orient=tk.HORIZONTAL, variable=self.smoothness_var,
                                     bg="#2b2b2b", fg="#ffffff", highlightthickness=0)
        smoothness_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.smoothness_label = ttk.Label(smoothness_frame, text="0.3", width=4)
        self.smoothness_label.pack(side=tk.LEFT, padx=5)
        
        def update_smoothness_label(*args):
            self.smoothness_label.config(text=f"{self.smoothness_var.get():.1f}")
        self.smoothness_var.trace('w', update_smoothness_label)
        
        self.tile_seamless_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="Make tileable/seamless (smooth borders)",
                       variable=self.tile_seamless_var).pack(anchor=tk.W, pady=5)
        
        ttk.Button(scrollable_frame, text="🎨 Generate Texture",
                  command=self.generate_texture, style="Accent.TButton").pack(fill=tk.X, pady=10)
        
        ttk.Label(scrollable_frame, text="Batch Generate:").pack(anchor=tk.W, pady=(10, 5))
        batch_frame = ttk.Frame(scrollable_frame)
        batch_frame.pack(fill=tk.X, pady=5)
        self.batch_count_var = tk.StringVar(value="5")
        ttk.Entry(batch_frame, textvariable=self.batch_count_var, width=8).pack(side=tk.LEFT)
        ttk.Label(batch_frame, text="variations").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(scrollable_frame, text="🔢 Batch Generate",
                  command=self.batch_generate).pack(fill=tk.X, pady=5)
    
    def create_preview_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="👁️ Preview & Colorize", padding="10")
        frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        canvas = tk.Canvas(frame, bg="#2b2b2b", highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        preview_frame = ttk.Frame(scrollable_frame)
        preview_frame.pack(pady=5)
        
        self.preview_canvas = tk.Canvas(preview_frame, width=256, height=256, 
                                       bg="#1e1e1e", highlightthickness=1)
        self.preview_canvas.pack()
        
        self.preview_label = ttk.Label(scrollable_frame, text="No texture generated yet")
        self.preview_label.pack(pady=5)
        
        ttk.Button(scrollable_frame, text="📂 Open in System Viewer",
                  command=self.open_last_texture).pack(pady=5, fill=tk.X)
        
        ttk.Label(scrollable_frame, text="Colorize Controls:").pack(pady=(10,5), anchor=tk.W)
        
        ttk.Label(scrollable_frame, text="Red:").pack(anchor=tk.W)
        self.red_var = tk.IntVar(value=128)
        red_slider = tk.Scale(scrollable_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                             variable=self.red_var, bg="#2b2b2b", fg="#ff0000", 
                             highlightthickness=0)
        red_slider.pack(fill=tk.X)
        
        ttk.Label(scrollable_frame, text="Green:").pack(anchor=tk.W)
        self.green_var = tk.IntVar(value=128)
        green_slider = tk.Scale(scrollable_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                               variable=self.green_var, bg="#2b2b2b", fg="#00ff00",
                               highlightthickness=0)
        green_slider.pack(fill=tk.X)
        
        ttk.Label(scrollable_frame, text="Blue:").pack(anchor=tk.W)
        self.blue_var = tk.IntVar(value=128)
        blue_slider = tk.Scale(scrollable_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                              variable=self.blue_var, bg="#2b2b2b", fg="#0000ff",
                              highlightthickness=0)
        blue_slider.pack(fill=tk.X)
        
        ttk.Button(scrollable_frame, text="🌈 Apply Color",
                  command=self.apply_custom_color, style="Accent.TButton").pack(fill=tk.X, pady=10)
        
        ttk.Label(scrollable_frame, text="Console Output:").pack(anchor=tk.W, pady=(10, 5))
        
        console_frame = ttk.Frame(scrollable_frame)
        console_frame.pack(fill=tk.BOTH, expand=True)
        
        self.console = scrolledtext.ScrolledText(console_frame, height=10, 
                                                 bg="#1e1e1e", fg="#00ff00",
                                                 font=("Consolas", 9))
        self.console.pack(fill=tk.BOTH, expand=True)
        
        self.log("🎮 Minecraft Texture AI Ready!")
        self.log("Add your 16x16 PNGs to training_data/ folders and start training!")
    
    def log(self, message):
        
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)
        self.console.update()
    
    def update_training_info(self):
        
        self.training_info.delete(1.0, tk.END)
        
        info = []
        info.append("📊 Training Data Status\n")
        info.append("="*30 + "\n\n")
        
        for texture_type in ['blocks', 'items']:
            data_dir = Path(__file__).parent / 'training_data' / texture_type
            if data_dir.exists():
                categories = [d.name for d in data_dir.iterdir() if d.is_dir()]
                if categories:
                    info.append(f"📦 {texture_type.upper()}:\n")
                    for cat in categories:
                        png_count = len(list((data_dir / cat).glob('*.png')))
                        info.append(f"  • {cat}: {png_count} PNGs\n")
                    info.append("\n")
                else:
                    info.append(f"⚠ {texture_type}: No categories\n\n")
            else:
                info.append(f"⚠ {texture_type}: Folder missing\n\n")
        
        self.training_info.insert(1.0, "".join(info))
    
    def set_texture_type(self, texture_type):
        
        self.current_texture_type = texture_type
        self.refresh_categories()
        self.log(f"Switched to: {texture_type}")
    
    def refresh_categories(self):
        
        data_dir = Path(__file__).parent / 'training_data' / self.current_texture_type
        
        if data_dir.exists():
            categories = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
            self.category_dropdown['values'] = categories
            if categories:
                self.category_dropdown.current(0)
                self.current_category = categories[0]
                self.log(f"Found {len(categories)} categories in {self.current_texture_type}")
            else:
                self.log(f"⚠ No categories found in {self.current_texture_type}")
        else:
            self.category_dropdown['values'] = []
            self.log(f"⚠ Directory not found: {data_dir}")
        
        self.update_training_info()
    
    def start_training(self):
        
        train_type = self.train_type_var.get()
        
        try:
            shading_epochs = int(self.shading_epochs.get())
            color_epochs = int(self.color_epochs.get())
        except ValueError:
            messagebox.showerror("Error", "Epochs must be numbers!")
            return
        
        self.log(f"\n{'='*50}")
        self.log(f"🚀 Starting training: {train_type}")
        self.log(f"Shading epochs: {shading_epochs}, Color epochs: {color_epochs}")
        self.log(f"{'='*50}\n")
        
        def train():
            try:
                if train_type == "all":
                    types = ["blocks", "items"]
                else:
                    types = [train_type]
                
                for texture_type in types:
                    self.log(f"\n📚 Training {texture_type}...")

                    self.log(f"  Stage 1: Shading model...")
                    self.ai.train_shading_model(texture_type, epochs=shading_epochs)

                    self.log(f"  Stage 2: Colorizer...")
                    self.ai.train_colorizer(texture_type, epochs=color_epochs)
                    
                    self.log(f"✅ {texture_type} training complete!\n")
                
                self.log(f"\n{'='*50}")
                self.log("🎉 ALL TRAINING COMPLETE!")
                self.log(f"{'='*50}\n")
                
                self.refresh_categories()
                messagebox.showinfo("Success", "Training complete! Ready to generate textures.")
                
            except Exception as e:
                self.log(f"❌ Error during training: {e}")
                messagebox.showerror("Training Error", str(e))
        
        thread = threading.Thread(target=train, daemon=True)
        thread.start()
    
    def generate_texture(self):
        category = self.category_var.get()
        output_name = self.output_name_var.get()
        seed_str = self.seed_var.get()
        smoothness = self.smoothness_var.get()
        tile_seamless = self.tile_seamless_var.get()
        
        if not category:
            messagebox.showwarning("No Category", "Please select a category!")
            return
        
        if not output_name:
            messagebox.showwarning("No Name", "Please enter an output name!")
            return
        
        seed = int(seed_str) if seed_str else None
        
        self.log(f"\n🎨 Generating {self.current_texture_type}/{category}: {output_name}")
        if seed:
            self.log(f"  Seed: {seed}")
        self.log(f"  Smoothness: {smoothness:.1f}")
        self.log(f"  Tileable: {'Yes' if tile_seamless else 'No'}")
        
        def generate():
            try:
                self.log("  Generating shading...")
                shaded_path = self.ai.generate_texture(
                    self.current_texture_type, 
                    category, 
                    output_name, 
                    seed=seed,
                    smoothness=smoothness,
                    tile_seamless=tile_seamless
                )
                self.log(f"  ✅ Created: {Path(shaded_path).name}")
                
                self.last_generated_path = shaded_path
                self.show_preview(shaded_path)
                
                self.log(f"✅ Generation complete!\n")
                messagebox.showinfo("Success", f"Texture generated!\n{Path(self.last_generated_path).name}")
                
            except Exception as e:
                self.log(f"❌ Error: {e}")
                messagebox.showerror("Generation Error", str(e))
        
        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
    
    def apply_custom_color(self):
        if not self.last_generated_path or not Path(self.last_generated_path).exists():
            messagebox.showwarning("No Texture", "Generate a texture first!")
            return
        
        r = self.red_var.get()
        g = self.green_var.get()
        b = self.blue_var.get()
        
        self.log(f"\n🌈 Applying color: R={r}, G={g}, B={b}")
        
        def colorize():
            try:
                img = Image.open(self.last_generated_path).convert('RGBA')
                pixels = np.array(img, dtype=np.float32)
                
                gray = pixels[:, :, 0]
                alpha = pixels[:, :, 3]
                
                intensity = gray / 255.0
                
                colored = np.zeros_like(pixels)
                colored[:, :, 0] = intensity * r
                colored[:, :, 1] = intensity * g
                colored[:, :, 2] = intensity * b
                colored[:, :, 3] = alpha
                
                colored = np.clip(colored, 0, 255).astype(np.uint8)
                
                output_name = Path(self.last_generated_path).stem.replace('_shaded', '') + '_colored'
                output_path = Path(self.last_generated_path).parent / f"{output_name}.png"
                
                Image.fromarray(colored).save(output_path)
                
                self.log(f"✅ Colored: {output_path.name}\n")
                self.last_generated_path = str(output_path)
                self.show_preview(str(output_path))
                messagebox.showinfo("Success", "Color applied!")
            except Exception as e:
                self.log(f"❌ Error: {e}")
                messagebox.showerror("Colorization Error", str(e))
        
        thread = threading.Thread(target=colorize, daemon=True)
        thread.start()
    
    def batch_generate(self):
        try:
            count = int(self.batch_count_var.get())
        except ValueError:
            messagebox.showerror("Error", "Count must be a number!")
            return
        
        if count < 1 or count > 50:
            messagebox.showerror("Error", "Count must be between 1 and 50!")
            return
        
        category = self.category_var.get()
        base_name = self.output_name_var.get()
        smoothness = self.smoothness_var.get()
        tile_seamless = self.tile_seamless_var.get()
        
        if not category or not base_name:
            messagebox.showwarning("Missing Info", "Please select category and enter output name!")
            return
        
        self.log(f"\n🔢 Batch generating {count} variations...")
        self.log(f"  Smoothness: {smoothness:.1f}, Tileable: {tile_seamless}")
        
        def batch():
            try:
                for i in range(count):
                    output_name = f"{base_name}_{i+1}"
                    self.log(f"  [{i+1}/{count}] Generating {output_name}...")
                    
                    self.ai.generate_texture(
                        self.current_texture_type,
                        category,
                        output_name,
                        seed=i,
                        smoothness=smoothness,
                        tile_seamless=tile_seamless
                    )
                
                self.log(f"✅ Batch complete! Generated {count} textures.\n")
                messagebox.showinfo("Success", f"Generated {count} textures!")
                
            except Exception as e:
                self.log(f"❌ Error: {e}")
                messagebox.showerror("Batch Error", str(e))
        
        thread = threading.Thread(target=batch, daemon=True)
        thread.start()
    
    def show_preview(self, image_path):
        
        try:
            img = Image.open(image_path)

            img_scaled = img.resize((256, 256), Image.NEAREST)
            photo = ImageTk.PhotoImage(img_scaled)
            
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(128, 128, image=photo)
            self.preview_canvas.image = photo  # Keep reference
            
            self.preview_label.config(text=f"Preview: {Path(image_path).name} (16x16 scaled to 256x256)")
        except Exception as e:
            self.log(f"⚠ Could not load preview: {e}")
    
    def open_last_texture(self):
        
        if self.last_generated_path and Path(self.last_generated_path).exists():
            import platform
            import subprocess
            
            system = platform.system()
            try:
                if system == "Windows":
                    os.startfile(self.last_generated_path)
                elif system == "Darwin":  # macOS
                    subprocess.run(["open", self.last_generated_path])
                else:  # Linux
                    subprocess.run(["xdg-open", self.last_generated_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {e}")
        else:
            messagebox.showwarning("No Texture", "Generate a texture first!")
    
    def open_training_folder(self):
        
        folder = Path(__file__).parent / 'training_data'
        folder.mkdir(exist_ok=True)
        
        import platform
        import subprocess
        
        system = platform.system()
        try:
            if system == "Windows":
                os.startfile(folder)
            elif system == "Darwin":
                subprocess.run(["open", folder])
            else:
                subprocess.run(["xdg-open", folder])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {e}")
    
    def open_output_folder(self):
        
        folder = Path(__file__).parent / 'output'
        folder.mkdir(exist_ok=True)
        
        import platform
        import subprocess
        
        system = platform.system()
        try:
            if system == "Windows":
                os.startfile(folder)
            elif system == "Darwin":
                subprocess.run(["open", folder])
            else:
                subprocess.run(["xdg-open", folder])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {e}")
    
    def show_quickstart(self):
        guide = "Quick Start: 1) Add PNGs to training_data folders 2) Train models 3) Generate textures"
        messagebox.showinfo("Quick Start Guide", guide)
    
    def show_about(self):
        about = "Minecraft 16x16 Texture AI Generator v1.0"
        messagebox.showinfo("About", about)

def main():
    
    root = tk.Tk()
    app = TextureAIGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()