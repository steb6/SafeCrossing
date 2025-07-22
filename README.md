# SafeCrossing

![Demo](demo.gif)

ACVSS25 Hackaton Project

## Installation
```
conda env create -f environment.yml
```
NOTE: to run on Windows with GPU, you have to download the right PyTorch version from https://pytorch.org/get-started/locally/

## Launch
```
conda activate SafeCrossing
```
```
ptyhon main.py
```
The first time that you launch the system, you need to set the requested point by directly clicking on the video.
You can adjust them at runtime by pressing 'a'.
To assess system performance, you need to label the data manually: press '+' to label subsequent frames as SAFE and press '-' to label subsequent frames as NOT_SAFE; the labels will be saved and used on the next run.