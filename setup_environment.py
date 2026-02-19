import subprocess
import sys
import platform

def install_dependencies():
    """
    Installs the required Python packages for the machine vision pipeline.
    """
    packages = [
        "tensorflow",   # Core ML library
        "numpy",        # Numerical computing
        "matplotlib",   # Plotting and visualization
        "pandas",       # Data manipulation (for CSV/Excel handling)
        "openpyxl",     # Excel export support
        "pillow",       # Image processing
        "scikit-learn", # ML metrics (added for completeness/future use)
    ]

    print("Detected platform:", platform.system(), platform.release())
    print(f"Installing {len(packages)} dependencies: {', '.join(packages)}")

    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}. Error: {e}")
            sys.exit(1)

    print("\nAll dependencies installed successfully.")
    print("You can now run the pipeline with: python main.py")

if __name__ == "__main__":
    install_dependencies()
