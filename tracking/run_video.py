import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation.tracker import Tracker


def run_video(tracker_name, tracker_param, input_video, output_video=None, bbox_path=None, debug=None, save_results=False):
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
    print(f"{tracker_name=}, {tracker_param=}, {input_video=}, {output_video=}, {bbox_path=}, {debug=}, {save_results=}")
    tracker = Tracker(tracker_name, tracker_param)
    tracker.run_video(input_video=input_video, output_video=output_video, bbox_path=bbox_path, debug=debug, save_results=save_results)

def main():
    parser = argparse.ArgumentParser(description='Run the tracker on a video.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('input_video', type=str, help='path to the input video.')
    parser.add_argument('--output', type=str, help='path to the saved video.')
    parser.add_argument('--bbox', type=str, help='path to the file containing bounding boxes with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    run_video(args.tracker_name, args.tracker_param, args.input_video, args.output, args.bbox, args.debug, args.save_results)


if __name__ == '__main__':
    main()
