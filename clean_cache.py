#!/usr/bin/env python3
"""
Clean cache script for pip and temporary files.
This helps free up disk space by removing unnecessary cache files.
"""

import os
import shutil
import sys
import subprocess

def get_pip_cache_dir():
    """Get the pip cache directory based on the OS"""
    if sys.platform == "win32":
        return os.path.expanduser("~\\AppData\\Local\\pip\\Cache")
    elif sys.platform == "darwin":  # macOS
        return os.path.expanduser("~/Library/Caches/pip")
    # Linux and other Unix-like systems
    return os.path.expanduser("~/.cache/pip")

def clean_pip_cache():
    """Clean pip cache directory"""
    cache_dir = get_pip_cache_dir()
    if os.path.exists(cache_dir):
        print(f"Removing pip cache from: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
            print("✓ Pip cache removed successfully")
        except Exception as e:
            print(f"Error removing pip cache: {e}")
    else:
        print("No pip cache found")

def clean_temp_files():
    """Clean temporary Python files"""
    # Clean __pycache__ directories in current workspace
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                print(f"Removing {pycache_path}")
                try:
                    shutil.rmtree(pycache_path)
                    print(f"✓ Removed {pycache_path}")
                except Exception as e:
                    print(f"Error removing {pycache_path}: {e}")

def clean_wheelhouse():
    """Clean any wheelhouse directories if they exist"""
    wheelhouse = "wheelhouse"
    if os.path.exists(wheelhouse):
        print(f"Removing wheelhouse directory: {wheelhouse}")
        try:
            shutil.rmtree(wheelhouse)
            print("✓ Wheelhouse directory removed successfully")
        except Exception as e:
            print(f"Error removing wheelhouse: {e}")

def run_pip_clean():
    """Run pip cache purge command (for newer pip versions)"""
    try:
        print("Running pip cache purge...")
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], check=True)
        print("✓ Pip cache purged successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running pip cache purge: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    print("Starting cleanup process...")
    
    # Clean pip cache
    clean_pip_cache()
    
    # Run pip cache purge (works for newer pip versions)
    run_pip_clean()
    
    # Clean temporary Python files
    clean_temp_files()
    
    # Clean wheelhouse if it exists
    clean_wheelhouse()
    
    print("\nCleanup completed!")
    print("\nTo reinstall dependencies:")
    print("For GPU version:")
    print("  pip install -r requirements.txt")
    print("For CPU version:")
    print("  pip install -r requirements_cpu.txt")

if __name__ == "__main__":
    main()