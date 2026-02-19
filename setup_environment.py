import subprocess
import sys
import platform

def check_python_version():
    """
    Checks if the current Python version is compatible with TensorFlow.
    TensorFlow typically requires Python 3.9-3.12 (64-bit).
    """
    version = sys.version_info
    major = version.major
    minor = version.minor

    # Check for 64-bit Python
    is_64bit = sys.maxsize > 2**32
    if not is_64bit:
        print("Error: TensorFlow requires a 64-bit Python installation.")
        print("Please install a 64-bit version of Python.")
        sys.exit(1)

    # Check version range (3.9 to 3.12 inclusive)
    if major == 3 and (9 <= minor <= 12):
        print(f"Python {major}.{minor} is compatible.")
        return True
    else:
        print(f"Error: Python {major}.{minor} is not supported by TensorFlow.")
        print("TensorFlow requires Python 3.9, 3.10, 3.11, or 3.12.")
        print("Please install a compatible Python version (e.g., Python 3.10 or 3.11).")
        sys.exit(1)

def install_dependencies():
    """
    Installs the required Python packages for the machine vision pipeline.
    """
    # Check Python version first
    check_python_version()

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
