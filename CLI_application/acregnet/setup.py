from setuptools import setup, find_packages

setup(
    name='acregnet',
    version='1.1',
    packages=find_packages('.'),
    include_package_data=True,
    install_requires=[
        'tensorflow==2.5.1',
        'numpy==1.14.5',
        'opencv-python==3.4.1.15'
        ],
    entry_points={
        'console_scripts': [
            'acregnet = acregnet.__main__:main'
        ]
    })
