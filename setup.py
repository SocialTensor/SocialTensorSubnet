import re
import os
import sys
import codecs
from os import path
from io import open
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

here = path.abspath(path.dirname(__file__))


class CustomInstallCommand(install):
    """Custom install command to run additional pip installs"""

    def run(self):
        # First, run the default install behavior
        print("Running standard install...")
        install.run(self)

        # Now, run the additional pip install commands sequentially
        try:
            # Install base requirements from requirements.txt
            print("Installing base requirements from requirements.txt...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], stdout=sys.stdout, stderr=sys.stderr)
            
            # Install controlnet-aux and pyiqa with --no-deps
            print("Installing controlnet-aux and pyiqa...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "controlnet-aux", "pyiqa", "--no-deps"], stdout=sys.stdout, stderr=sys.stderr)
            
            # Install onnxruntime-gpu with extra index URL for CUDA support
            print("Installing onnxruntime-gpu...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime-gpu", "--extra-index-url", "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"], stdout=sys.stdout, stderr=sys.stderr)

            # Optional: Remove or adjust LD_LIBRARY_PATH setting for your environment
            print("Setting LD_LIBRARY_PATH...")
            subprocess.check_call(['bash', '-c', 'export LD_LIBRARY_PATH=$(pwd)/venv/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH'], stdout=sys.stdout, stderr=sys.stderr)
            
        except subprocess.CalledProcessError as e:
            print(f"Error during the installation process: {e}")
            sys.exit(1)


class CustomDevelopCommand(develop):
    """Custom develop command to run additional pip install --editable / -e"""

    def run(self):
        # First, run the default develop behavior
        print("Running standard pip install -e...")
        develop.run(self)

        # Now, run the additional pip develop commands sequentially
        try:
            # Install base requirements from requirements.txt
            print("Installing base requirements from requirements.txt...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], stdout=sys.stdout, stderr=sys.stderr)
            
            # Install controlnet-aux and pyiqa with --no-deps
            print("Installing controlnet-aux and pyiqa...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "controlnet-aux", "pyiqa", "--no-deps"], stdout=sys.stdout, stderr=sys.stderr)
            
            # Install onnxruntime-gpu with extra index URL for CUDA support
            print("Installing onnxruntime-gpu...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime-gpu", "--extra-index-url", "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"], stdout=sys.stdout, stderr=sys.stderr)

            # Optional: Remove or adjust LD_LIBRARY_PATH setting for your environment
            print("Setting LD_LIBRARY_PATH...")
            subprocess.check_call(['bash', '-c', 'export LD_LIBRARY_PATH=$(pwd)/venv/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH'], stdout=sys.stdout, stderr=sys.stderr)
            
        except subprocess.CalledProcessError as e:
            print(f"Error during the installation process: {e}")
            sys.exit(1)


with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# loading version from setup.py
with codecs.open(
    os.path.join(here, "image_generation_subnet/__init__.py"), encoding="utf-8"
) as init_file:
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M
    )
    version_string = version_match.group(1)

print('gauhasiufhsaifhsduifhuds')

setup(
    name="image_generation_subnet",
    version=version_string,
    description="nicheimage_subnet for image generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NicheTensor/NicheImage",
    author="bittensor.com",
    packages=find_packages(),
    include_package_data=True,
    author_email="",
    license="MIT",
    python_requires=">=3.11",
    install_requires=[],  # No deps here, as we'll handle them manually
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    cmdclass={
        "install": CustomInstallCommand,
        "develop": CustomDevelopCommand
    },
)
