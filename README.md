# ARTrack Fork

This is a fork of [MIV-XJTU/ARTrack](https://github.com/MIV-XJTU/ARTrack).

This repository is being used under the Apache License 2.0 for commercial use. The code, including modifications and distributions, is licensed under this open-source license, which allows for commercial use, modification, and distribution.

See the `LICENSE` file for more details.

## Purpose

The primary purpose of this code is to export a model to ONNX format so that it can be used in Unity through the Unity Sentis package for inference and tracking. These models are designed for **visual object tracking**, enabling the tracking of objects in video sequences based on their appearance.

## Modifications and Additions

### New files in `tracking/`
- `run_video.py`
- `export2onnx.py`
- `run_onnx.py`

### Tracker changes
- Modified the ARTrackV2 tracker code to support model export.

### ONNX Model Support
- Added a `Dockerfile` to run the model (requires an NVIDIA GPU).

## Setting Up and Running in Docker

### 1. Download the Checkpoints
Before building the Docker image, you need to download the required model checkpoints. You can get the checkpoints from one of the following locations:

- [Google Drive link](https://drive.google.com/file/d/1tGaY5jQxZOTzJDWXgOgoHtBwc5l4NLQ2/view)
- Or, directly from the original [ARTrack repository](https://github.com/MIV-XJTU/ARTrack)

### 2. Rename the Checkpoints
After downloading the checkpoint files, rename them to `artrackV2_seq_256_full` and move them to the same folder as the `Dockerfile`.


### 3. Build the Docker Image
Navigate to the directory containing the `Dockerfile` and build the image with the following command:

```sh
docker build -t artrack_image .
```

### 4. Run the Docker Container
Launch the container with
```sh
docker run -it artrack_image
```

### 5. Inside the Container Terminal
Once inside the container terminal, initialize Conda and activate the environment:
```sh
conda init
source ~/.bashrc
conda activate artrack
```
Then, create the default local file structure by running:
```sh
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

## Running the Model
### Run a video with the model
You can run the model on any video by specifying the input video, output file, and initial bounding box. Here's the general command format:
```sh
python tracking/run_video.py artrackv2_seq_256_full <input_video> <output_video> <initial_bounding_box> [--save_results]
```
- `<input_video>`: Provide the path to your input video.
- `<output_video>`: Specify the name of the output video file.
- `<initial_bounding_box>`: Provide the initial bounding box in the format `"x,y,width,height"`.
- `[--save_results]`: Optional flag to save results.

### Export the Model to ONNX
To export the model to ONNX format, you need to provide a video. The video is used to generate dummy inputs for the model export. Here's the command:
```sh
python tracking/export2onnx.py artrackv2_seq_256_full <input_video>
```
- `<input_video>`: Provide the path to your input video, which will be used to generate dummy inputs for the export.

### Run the ONNX model
To run the model using ONNX, you can use different execution providers (CPU or GPU). By default, it runs on CUDA (NVIDIA GPU required):
```sh
python tracking/run_onnx.py artrackv2_seq_256_full <input_video> <output_onnx_file>
```
- `<input_video>`: Provide the path to your input video.
- `<output_onnx_file>`: Specify the path for the output ONNX file.
