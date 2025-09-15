#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import argparse
import platform
import shutil

py_version = sys.version
print(py_version)

# Get the correct Python executable for Windows
python_exe = sys.executable
print(f"Using Python executable: {python_exe}")

file = os.getcwd() 
print(f"Current directory: {file}")

# Windows-compatible version update (skip shell script)
print("Updating version file...")
print('*'*100)

# Git operations
try:
    result = subprocess.run(["git", "pull"], check=True, capture_output=True, text=True)
    print('git pull done')
    print('*'*100)
except subprocess.CalledProcessError as e:
    print(f"Git pull failed: {e}")

try:
    result = subprocess.run(["git", "push"], check=True, capture_output=True, text=True)
    print('git push done')
except subprocess.CalledProcessError as e:
    print(f"Git push failed: {e}")

print('*'*100)
print('Removing dist and build folders')

# Remove dist and build directories if they exist (Windows compatible)
dist_path = os.path.join(file, 'dist')
build_path = os.path.join(file, 'build')

if os.path.exists(dist_path):
    shutil.rmtree(dist_path)
    print(f"Removed {dist_path}")

if os.path.exists(build_path):
    shutil.rmtree(build_path)
    print(f"Removed {build_path}")

# List directory contents
print("Directory contents:")
for item in os.listdir(file):
    print(f"  {item}")

# Build the package
print('*'*100)
print('Building package...')
try:
    result = subprocess.run([python_exe, "-m", "build"], check=True, capture_output=True, text=True, cwd=file)
    print('Package built successfully')
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Build failed: {e}")
    print(f"Error output: {e.stderr}")
    # Try to install build if it's missing
    print("Attempting to install build module...")
    subprocess.run([python_exe, "-m", "pip", "install", "build"], check=True)
    # Retry build
    result = subprocess.run([python_exe, "-m", "build"], check=True, capture_output=True, text=True, cwd=file)
    print('Package built successfully after installing build module')

print('*'*100)
print('Wheel built')

# Install the built package
dist_files = os.listdir(os.path.join(file, 'dist'))
wheel_files = [f for f in dist_files if f.endswith('.whl')]

if wheel_files:
    wheel_file = wheel_files[-1]  # Get the latest wheel file
    wheel_path = os.path.join(file, 'dist', wheel_file)
    
    install_cmd = [python_exe, "-m", "pip", "install", wheel_path, "--force-reinstall"]
    print(f"Installing: {' '.join(install_cmd)}")
    
    try:
        result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
        print('Package installed successfully')
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        print(f"Error output: {e.stderr}")
else:
    print("No wheel files found in dist directory")

print('Package installation completed')
print('*'*100)

# Optional: Upload to PyPI (commented out for safety)
# print('Uploading to PyPI...')
# try:
#     result = subprocess.run([python_exe, "-m", "twine", "upload", "dist/*"], check=True, capture_output=True, text=True, cwd=file)
#     print('Upload to PyPI completed')
# except subprocess.CalledProcessError as e:
#     print(f"Upload failed: {e}")
#     print("Make sure twine is installed: pip install twine")
