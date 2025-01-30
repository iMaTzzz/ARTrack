import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation.tracker import Tracker


def export2onnx(tracker_name, tracker_param, input_video, input_onnx):
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
    print(f"{tracker_name=}, {tracker_param=}, {input_video=}, {input_onnx=}")

    # Initializing tracker
    tracker = Tracker(tracker_name, tracker_param)

    init_bbox = [672, 108, 576, 864]

    tracker.run_onnx(input_video=input_video, init_bbox=init_bbox, input_onnx=input_onnx)
    # torch.onnx.dynamo_export(tracker)

def main():
    parser = argparse.ArgumentParser(description='Run the tracker on a video.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('input_video', type=str, help='path to the input video.')
    parser.add_argument('input_onnx', type=str, help='path to the onnx model.')

    args = parser.parse_args()

    export2onnx(args.tracker_name, args.tracker_param, args.input_video, args.input_onnx)


if __name__ == '__main__':
    main()

