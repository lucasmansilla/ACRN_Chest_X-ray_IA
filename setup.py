from setuptools import find_packages, setup

setup(
    name='acregnet',
    version='0.1',
    description='Learning Deformable Registration of Medical Images with Anatomical Constraints',
    url='https://github.com/lucasmansilla/ACRN_Chest_X-ray_IA',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'numpy',
        'medpy',
        'pillow'
    ],
    entry_points={
        'console_scripts': [
            'acregnet=acregnet.__main__:main',
        ],
    }
)
