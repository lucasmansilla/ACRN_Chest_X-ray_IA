# AC-RegNet CLI Application
An open source command line tool for chest X-ray image registration using a pre-trained AC-RegNet model.

## Requirements
To run the application, you need to install Python 3 and the following libraries: 
- [TensorFlow](https://www.tensorflow.org/) (1.5.0)
- [NumPy](http://www.numpy.org/) (1.14.5)
- [OpenCV](https://opencv.org/) (3.4.1.15)
- [MedPy](https://loli.github.io/medpy/) (0.4.0)

These modules are not included in the standard Python library, but are listed in the program dependencies list and will be automatically installed through [pip](https://pip.pypa.io/en/stable/) when the program is installed.

## Install
In the root directory of the application, run the file **install.sh**:
```
./install.sh
```
This will install the application automatically using [pip](https://pip.pypa.io/en/stable/).

## Usage
```
acregnet register <target> <source> --dest=<destination-directory>
```

#### Description:
- *acregnet*: name of the program.
- *register*: command to register the input images.
- *&lt;target&gt;*: path to the target image.
- *&lt;source&gt;*: path to the source image. 
- *&lt;destination-directory&gt;*: path to the directory where the resulting image and deformation field will be stored.

The input images must be PNG, square and equal in size.

#### Example
```
user@host:~$ acregnet register image001.png image002.png --dest=/home/user/my_dir
Loading input images...done
Building AC-RegNet model...done
Loading trained AC-RegNet model...done
Registering images...
Result image and deformation field saved in /home/user/my_dir
```

## Uninstall
```
pip uninstall acregnet
```

## Program version
1.1

## Author
Lucas Andrés Mansilla (lucasmansilla12@gmail.com)
