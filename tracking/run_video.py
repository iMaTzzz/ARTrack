import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation.tracker import Tracker


import argparse

def parse_bbox(bbox_str):
    """Parse the bounding box input as x,y,w,h."""
    try:
        bbox = list(map(float, bbox_str.split(',')))
    except ValueError:
        raise ValueError(f"Invalid bounding box format: {bbox_str}. Expected format: x,y,w,h.")

    if len(bbox) != 4:
        raise ValueError(f"Bounding box must have exactly 4 values, got {len(bbox)}: {bbox}")
    return bbox


def run_video(tracker_param, input_video, output_video, bbox, debug=None, save_results=False):
    """Run the tracker on a video.
    
    Args:
        tracker_param: Name of the tracker configuration.
        input_video: Path to the input video (mp4).
        output_video: Path to the output video (mp4).
        bbox: Bounding box as x,y,w,h.
        debug: Debug level.
        save_results: Whether to save the predictions.
    """
    print(f"{input_video=}, {output_video=}, {bbox=}, {debug=}, {save_results=}")
    
    # Parse the bounding box
    parsed_bbox = parse_bbox(bbox)
    print(f"Parsed bounding box: {parsed_bbox}")
    
    # Initialize and run the tracker
    tracker = Tracker("artrackv2_seq", tracker_param)
    tracker.run_video(
        input_video=input_video,
        output_video=output_video,
        bbox_path=parsed_bbox,
        debug=debug,
        save_results=save_results
    )


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on a video.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('input_video', type=str, help='Path to the input video.')
    parser.add_argument('output', type=str, help='Path to the saved video.')
    parser.add_argument('bbox', type=str, help='Bounding box as x,y,w,h (comma-separated).')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    run_video(
        tracker_param=args.tracker_param,
        input_video=args.input_video,
        output_video=args.output,
        bbox=args.bbox,
        debug=args.debug,
        save_results=args.save_results
    )


if __name__ == '__main__':
    main()
