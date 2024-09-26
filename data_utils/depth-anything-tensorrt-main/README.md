<div align="center">

Depth Anything TensorRT
===========================

[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-11.6-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-8.6-green)](https://developer.nvidia.com/tensorrt)
[![mit](https://img.shields.io/badge/license-MIT-blue)](https://github.com/spacewalk01/depth-anything-tensorrt/blob/main/LICENSE)

</div>

Depth estimation is the task of measuring the distance of each pixel relative to the camera. This repo provides a TensorRT implementation of the [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) depth estimation model in both C++ and Python, enabling efficient real-time inference.

<p align="center">
  <img src="assets/davis_dolphins_result.gif" height="225px" width="720px" />
</p>

## ⏱️ Performance

The inference time includes the pre-preprocessing and post-processing stages:
| Device          | Model | Model Input (WxH) |  Image Resolution (WxH)|Inference Time(ms)|
|:---------------:|:------------:|:------------:|:------------:|:------------:|
| RTX4090        | Depth-Anything-S  |518x518  |  1280x720    | 3     |
| RTX4090        | Depth-Anything-B  |518x518  |  1280x720    | 6     |
| RTX4090        | Depth-Anything-L  |518x518  |  1280x720    | 12    |


> [!NOTE]
> Inference was conducted using `FP16` precision, with a warm-up period of 10 frames. The reported time corresponds to the last inference.

## 🚀 Quick Start

#### C++

- **Usage 1**: Create an engine from an onnx model and save it:
``` shell
depth-anything-tensorrt.exe <onnx model> <input image or video>
```
- **Usage 2**: Deserialize an engine. Once you've built your engine, the next time you run it, simply use your engine file:
``` shell
depth-anything-tensorrt.exe <engine> <input image or video>
```

Example:
``` shell
# infer image
depth-anything-tensorrt.exe depth_anything_vitb14.engine test.jpg
# infer folder(images)
depth-anything-tensorrt.exe depth_anything_vitb14.engine data
# infer video
depth-anything-tensorrt.exe depth_anything_vitb14.engine test.mp4 # the video path
```

#### Python

```
cd depth-anything-tensorrt/python

# infer image
python trt_infer.py --engine <path to trt engine> --img <single-img> --outdir <outdir> [--grayscale]
```

## 🛠️ Build

#### C++

Refer to our [docs/INSTALL.md](https://github.com/spacewalk01/depth-anything-tensorrt/blob/main/docs/INSTALL.md) for C++ environment installation.

#### Python

``` shell
cd <tensorrt installation path>/python
pip install cuda-python
pip install tensorrt-8.6.0-cp310-none-win_amd64.whl
pip install opencv-python
``` 

## 🤖 Model Preparation

Perform the following steps to create an onnx model:

1. Download the pretrained [model](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints) and install [Depth-Anything](https://github.com/LiheYoung/Depth-Anything):
   ``` shell
   git clone https://github.com/LiheYoung/Depth-Anything
   cd Depth-Anything
   pip install -r requirements.txt
   ```

2. Copy and paste [dpt.py](https://github.com/spacewalk01/depth-anything-tensorrt/blob/main/dpt.py) in this repo to `<depth_anything_installpath>/depth_anything` folder. Then, copy [export.py](https://github.com/spacewalk01/depth-anything-tensorrt/blob/main/export.py) in this repo to `<depth_anything_installpath>`. Note that I've only removed a squeeze operation at the end of model's forward function in `dpt.py` to avoid conflicts with TensorRT.
3. Export the model to onnx format using [export.py](https://github.com/spacewalk01/depth-anything-tensorrt/blob/main/export.py). You will get an onnx file named `depth_anything_vit{}14.onnx`, such as `depth_anything_vitb14.onnx`. Note that I used torch cpu version for exporting the onnx model as it is not necessary to deploy the model on GPU when exporting.

    
    ``` shell
    conda create -n depth-anything python=3.8
    conda activate depth-anything
    pip install torch torchvision
    pip install opencv-python
    pip install onnx
    python export.py --encoder vitb --load_from depth_anything_vitb14.pth --image_shape 3 518 518
    ```

  > [!TIP]
  > The width and height of the model input should be divisible by 14, the patch height.

4. Finally, copy the opencv dll files such as `opencv_world490.dll` and `opencv_videoio_ffmpeg490_64.dll` into the `<depth_anything_installpath>/build/Release` folder.

## 👏 Acknowledgement

This project is based on the following projects:
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) - Unleashing the Power of Large-Scale Unlabeled Data.
- [TensorRT](https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples) - TensorRT samples and api documentation.
