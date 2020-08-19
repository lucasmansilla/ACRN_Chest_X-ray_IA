# Learning Deformable Registration of Medical Images with Anatomical Constraints
Repository of AC-RegNet, a new method to regularize CNN-based deformable image registration by considering global anatomical priors in the form of segmentation masks.

In this repository you can find the source code corresponding to the paper "Learning deformable registration of medical images with anatomical constraints" (Neural Networks, 2020). You can download our paper from [here](https://arxiv.org/abs/2001.07183).

## Content
- [CLI Application](https://github.com/lucasmansilla/ACRN_Chest_X-ray_IA/tree/master/CLI_application/acregnet): An open source command line tool for chest x-ray image registration.
- [AC-RegNet](https://github.com/lucasmansilla/ACRN_Chest_X-ray_IA/tree/master/acregnet): Implementation of AC-RegNet with TensorFlow.
- [NIH Chest-XRay14 segmentations](https://github.com/lucasmansilla/NIH_chest_xray14_segmentations): Anatomical segmentation masks produced for the NIH Chest-XRay14 dataset using AC-RegNet and a multi-atlas segmentation model.

## Reference
- Mansilla, L., Milone, D. H., & Ferrante, E. (2020). Learning deformable registration of medical images with anatomical constraints. Neural Networks, 124, 269-279.

## License
[MIT](https://choosealicense.com/licenses/mit/)
