# Learning Deformable Registration of Medical Images with Anatomical Constraints
This repository contains an extended version of the source code corresponding to the paper "Learning deformable registration of medical images with anatomical constraints" (Neural Networks, 2020). You can check out our paper from here: https://arxiv.org/abs/2001.07183.

## Instructions
This project uses Python 3.8.10 and PyTorch 1.10.1.

### Project environment:
1. Create and activate virtual environment: 1) `python3 -m venv env`, 2) `source env/bin/activate`
2. Install required packages: `pip install -r requirements.txt`
3. Install project modules (acregnet): `pip install -e .`

### Simulations:
- AENet: `./01_run_aenet.sh`
- AC-RegNet: `./02_run_acregnet.sh`
In both scenarios, by editing the variable named `DATASET` you can choose the input dataset: JSRT, Montgomery or Shenzhen.

## AC-RegNet CLI Application
A command line tool for chest x-ray image registration is also provided. Here's an example of how to use it:
```
acregnet --mov JPCLN001.png --fix JPCLN002.png --model results/JSRT/ACRegNet/train/model.pt --dst output
```

## NIH Chest-XRay14 segmentations
Anatomical segmentation masks produced with a Multi-atlas segmentation model and AC-RegNet: https://github.com/lucasmansilla/NIH_chest_xray14_segmentations.


## Reference
- Mansilla, L., Milone, D. H., & Ferrante, E. (2020). Learning deformable registration of medical images with anatomical constraints. Neural Networks, 124, 269-279.

## License
[MIT](https://choosealicense.com/licenses/mit/)
