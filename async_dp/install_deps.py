"""
Async DP Dependency Installer
Requires: Python 3.12+, uv package manager
"""
import subprocess
import sys
import shutil


def check_python_version():
    """Check Python version is 3.12+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 12):
        print(f"[Error] Python 3.12+ required. Current: {version.major}.{version.minor}.{version.micro}")
        print("Install Python 3.12: https://www.python.org/downloads/")
        sys.exit(1)
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")


def check_uv_installed():
    """Check uv package manager is installed"""
    if shutil.which("uv") is None:
        print("[Error] 'uv' package manager not found.")
        print("Install uv: https://github.com/astral-sh/uv")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("  or: pip install uv")
        sys.exit(1)

    result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
    version = result.stdout.strip() if result.returncode == 0 else "unknown"
    print(f"[OK] uv {version}")


def run_uv(args, desc):
    """Run uv command with error handling"""
    print(f"\n[Installing] {desc}")
    try:
        subprocess.check_call(["uv"] + args)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Failed to install {desc}: {e}")
        sys.exit(1)


def main():
    print("=" * 50)
    print("  Async DP Dependency Installer")
    print("=" * 50)

    # 1. Check requirements
    print("\n[Checking Requirements]")
    check_python_version()
    check_uv_installed()

    # 2. Sync dependencies from pyproject.toml
    print("\n[Installing Dependencies]")
    run_uv(["sync"], "Core dependencies from pyproject.toml")

    # 3. Install Interbotix SDK (optional, may fail without ROS)
    print("\n[Installing Optional: Interbotix SDK]")
    try:
        url = "git+https://github.com/Interbotix/interbotix_ros_toolboxes.git#subdirectory=interbotix_xs_toolbox/interbotix_xs_modules"
        subprocess.check_call(["uv", "pip", "install", url, "--no-build-isolation"])
        print("[OK] Interbotix SDK installed")
    except subprocess.CalledProcessError:
        print("[Skip] Interbotix SDK (requires ROS dependencies)")

    print("\n" + "=" * 50)
    print("  Installation Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("  uv run pytest          # Run tests")
    print("  uv run python main.py --mode train  # Train model")
    print("  uv run python scripts/run_simulation.py  # Run simulation")


if __name__ == "__main__":
    main()
