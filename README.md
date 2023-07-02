# Implementation of SSD in PyTorch to Classify Pavement Distress For Video

> Please make sure you have Nvidia CUDA installed on your system.

This repository implements [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325). This implementation is heavily influenced by the projects [high quality, fast, modular reference implementation of ssd in pytorch](https://github.com/lufficc/SSD), this repository aims to classify pavement distress based on SSD using video as an input.

### Step-by-step installation

```bash
git clone https://github.com/bruhtus/pavement_distress_ssd.git
cd pavement_distress_ssd
# Required packages:
pip install -r requirements.txt
```


## Train

### Setting Up Datasets

For COCO dataset, make the folder structure like this:
```
ssd/data/datasets
|__ annotations
    |_ train.json
    |_ ...
|__ train
    |_ <im-1-name>.jpg
    |_ <im-1-name>.json
    |_ ...
    |_ <im-N-name>.jpg
    |_ <im-N-name>.json
|__ ...
```
Use labelme to do labeling stuff and then use [labelme2coco.py](https://github.com/Tony607/labelme2coco) to generate COCO data formatted JSON.

## Test
Please see [documentation.md](documentation.md) for more detailed usage of the testing implementation.

## Develop Guide

If you want to add your custom components, please see [DEVELOP GUIDE on lufficc repo](https://github.com/lufficc/SSD/blob/master/DEVELOP_GUIDE.md) for more details.

## Citations
If you use this project in your research, please cite this project.
```text
@misc{bruhtus2020,
    author = {Robertus Diawan Chris},
    title = {{Implementation of SSD in PyTorch to Classify Pavement Distress on Video},
    year = {2020},
    howpublished = {\url{https://github.com/bruhtus/pavement_distress_ssd}}
}
```
