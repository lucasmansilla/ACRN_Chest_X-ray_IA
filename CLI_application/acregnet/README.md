# AC-RegNet CLI Application
An open source command line tool for chest X-ray image registration using a pre-trained AC-RegNet model.

## Requirements
To run the application you need the **Python 2.7** interpreter and the following modules: 
- [TensorFlow](https://www.tensorflow.org/) (tensorflow&gt;=1.5.0)
- [NumPy](http://www.numpy.org/) (numpy&gt;=1.14.5)
- [OpenCV](https://opencv.org/)(opencv-python&gt;=3.4.1.15)
- [MedPy](https://loli.github.io/medpy/)(medpy==0.3.0)

These modules are not part of the standard Python library, but are included in the program dependencies list and will be automatically installed through [pip](https://pip.pypa.io/en/stable/) when the program is installed.

## Install
In the root directory of the source code, run the file **install.sh**:
```
./install.sh
```
This will install the application automatically using [pip](https://pip.pypa.io/en/stable/).

## Usage
```
acregnet register <target> <source> --dest=<destination-directory>>
```

#### Description:
- *acregnet*: name of the program.
- *register*: command to register the input images.
- *&lt;target&gt;*: path to the target image.
- *&lt;source&gt;*: path to the source image. 
- *&lt;destination-directory&gt;*: path to the directory where the resulting image and the deformation field information will be stored.

Input images must be square PNG images of the same size.

#### Example
Using two example images extracted from the [JSRT](http://db.jsrt.or.jp/eng.php) dataset:
```
user@host:~$ acregnet register JPCLN075.png JPCLN117.png --dest=/home/user/my_dir
Loading input images...done
Building AC-RegNet model...done
Loading trained AC-RegNet model...done
Registering images...
Result image and deformation field information saved in: /home/user/my_dir
```

## Uninstall
```
pip uninstall acregnet
```

## Program version
0.1.0

## Author
Lucas Andrés Mansilla (lucasmansilla12@gmail.com)
