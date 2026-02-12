"""
Automatic Installer for Minecraft Texture AI
Installs all required dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_command(command, description):
    """Run a command and show progress"""
    print(f"⏳ {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"✓ {description} complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def install_dependencies():
    """Install all required Python packages"""
    print_header("MINECRAFT TEXTURE AI - AUTOMATIC INSTALLER")
    
    print("This will install the following packages:")
    print("  • PyTorch (CPU version)")
    print("  • NumPy")
    print("  • Pillow (PIL)")
    print("  • Tkinter (if not already installed)")
    print()
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print()
    
    # Upgrade pip first
    if not run_command(
        f'"{sys.executable}" -m pip install --upgrade pip',
        "Upgrading pip"
    ):
        print("Warning: Could not upgrade pip, continuing anyway...")
    
    # Install PyTorch CPU version (smaller, faster for laptops)
    print_header("Installing PyTorch (CPU version)")
    if not run_command(
        f'"{sys.executable}" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu',
        "Installing PyTorch"
    ):
        print("❌ Failed to install PyTorch!")
        print("Trying alternative method...")
        if not run_command(
            f'"{sys.executable}" -m pip install torch torchvision',
            "Installing PyTorch (alternative)"
        ):
            return False
    
    # Install other dependencies
    print_header("Installing other dependencies")
    
    packages = [
        ("numpy", "NumPy (numerical computing)"),
        ("Pillow", "Pillow (image processing)")
    ]
    
    for package, description in packages:
        if not run_command(
            f'"{sys.executable}" -m pip install {package}',
            f"Installing {description}"
        ):
            print(f"❌ Failed to install {package}")
            return False
    
    # Test imports
    print_header("Testing installations")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch import failed!")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        print("❌ NumPy import failed!")
        return False
    
    try:
        import PIL
        print(f"✓ Pillow {PIL.__version__}")
    except ImportError:
        print("❌ Pillow import failed!")
        return False
    
    try:
        import tkinter
        print("✓ Tkinter (GUI)")
    except ImportError:
        print("⚠ Tkinter not available - GUI may not work")
        print("  On Windows, Tkinter should be included with Python")
        print("  On Linux: sudo apt-get install python3-tk")
        print("  On Mac: It should be included, try reinstalling Python")
    
    # Create folder structure
    print_header("Setting up folder structure")
    
    base = Path(__file__).parent / 'training_data'
    
    # Block categories
    block_cats = ['wood', 'metal', 'stone', 'ore', 'glass', 'natural', 'decorative']
    blocks_dir = base / 'blocks'
    blocks_dir.mkdir(parents=True, exist_ok=True)
    for cat in block_cats:
        (blocks_dir / cat).mkdir(exist_ok=True)
    
    # Item categories  
    item_cats = ['sword', 'axe', 'pickaxe', 'shovel', 'tool', 'armor', 'food', 'material']
    items_dir = base / 'items'
    items_dir.mkdir(parents=True, exist_ok=True)
    for cat in item_cats:
        (items_dir / cat).mkdir(exist_ok=True)
    
    # Create other directories
    (Path(__file__).parent / 'models').mkdir(exist_ok=True)
    (Path(__file__).parent / 'output').mkdir(exist_ok=True)
    
    print("✓ Created folder structure!")
    print(f"  📁 {blocks_dir}/")
    for cat in block_cats:
        print(f"     └─ {cat}/")
    print(f"  📁 {items_dir}/")
    for cat in item_cats:
        print(f"     └─ {cat}/")
    
    print_header("INSTALLATION COMPLETE!")
    
    print("✓ All dependencies installed successfully!")
    print("✓ Folder structure created!")
    print()
    print("Next steps:")
    print("  1. Add your 16x16 PNG textures to training_data/blocks/ or items/ subfolders")
    print("  2. Double-click 'run_gui.py' to open the GUI")
    print("  3. Click 'Train' to train the AI on your textures")
    print("  4. Click 'Generate' to create new textures!")
    print()
    input("Press Enter to exit...")
    
    return True

if __name__ == "__main__":
    try:
        success = install_dependencies()
        if not success:
            print("\n❌ Installation failed!")
            print("Please check the error messages above.")
            input("Press Enter to exit...")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)