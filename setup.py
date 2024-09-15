from setuptools import setup
from setuptools import find_packages

exclude_dirs = ("configs",)

# for install, do: pip install -ve .

setup(
    name='latte',
    version="0.0.1",
    url="https://github.com/AronCao49/Latte",
    description="Reliable Spatial-Temporal Voxels for Multi-Modal Test-Time Adaptation",
    install_requires=[
        'yacs', 
        'nuscenes-devkit', 
        'tabulate', 
        'opencv-python==4.5.5.64', 
        'Werkzeug==2.2.2',
        'timm==0.4.12',
        'openpyxl',
        'imageio'],
    packages=find_packages(exclude=exclude_dirs),
)