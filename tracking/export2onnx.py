import os
import sys
import argparse
import cv2 as cv
import torch

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation.tracker import Tracker


def export2onnx(tracker_name, tracker_param, input_video):
    """Run the tracker on a video.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        input_video: Path to the input video (mp4)
        output: Path to the output video (mp4)
        optional_bbox: Path to the bounding boxes (txt)
        debug: Debug level.
        save_results: Bool if we want to save the predictions
    """
    print(f"{tracker_name=}, {tracker_param=}, {input_video=}")
    cap = cv.VideoCapture(input_video)
    # Get video properties
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    success, frame = cap.read()
    print(f"frame{frame=}")
    # beach.mp4
    init_bbox = torch.tensor([790, 1440, 171, 387])
    tracker = Tracker(tracker_name, tracker_param)
    tracker.export2onnx(input_video=input_video, init_bbox=init_bbox)

    input_shape = (1, 3, width, height)
    onnx_path = "tracking.onnx"
    input_names = ['input']
    output_names = ['bbox_pred']
    dynamic_axes = {'input': {2: 'width', 3: 'height'}}
    # torch.onnx.export(tracker, input_shape, onnx_path, input_names, output_names, dynamic_axes)

    # torch.onnx.dynamo_export(tracker)

def main():
    parser = argparse.ArgumentParser(description='Run the tracker on a video.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('input_video', type=str, help='path to the input video.')

    args = parser.parse_args()

    export2onnx(args.tracker_name, args.tracker_param, args.input_video)


if __name__ == '__main__':
    main()
