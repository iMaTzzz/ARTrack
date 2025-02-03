import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import torch
import onnxruntime
import onnx
from lib.train.data.processing_utils import sample_target, transform_image_to_crop


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, display_name: str = None):
        self.name = name
        self.parameter_name = parameter_name
        self.display_name = display_name

        env = env_settings()
        self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, input_video, output_video, bbox_path, debug=None, save_results=False):
        """Run the tracker with the videofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
        else:
            raise ValueError(f"Unknown multi object mode {multiobj_mode}")

        # Ensure input video is valid
        assert os.path.isfile(input_video), f"Invalid input video file: {input_video}"
        assert bbox_path is not None and len(bbox_path) == 4, "bbox_path must be a valid list or tuple in [x, y, w, h] format."

        # Prepare the output list for bounding boxes
        output_boxes = []

        # Open video file
        cap = cv.VideoCapture(input_video)
        # Get video properties
        fps = cap.get(cv.CAP_PROP_FPS)
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        # Start writing video output
        if output_video is not None:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
            output = cv.VideoWriter(output_video, fourcc, fps, (width, height))

        # Initialize tracker with the first frame and bounding box
        success, frame = cap.read()
        if not success:
            print(f"Failed to read the first frame from {input_video}.")
            return

        def _build_init_info(box):
            return {'init_bbox': box}

        tracker.initialize(frame, _build_init_info(bbox_path))
        output_boxes.append(bbox_path)

        # Process the video frame by frame
        frame_number = 1
        total_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if no more frames are available
            frame_number += 1

            # Track the object
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)
            inference_time = out['time']
            total_time += inference_time
            print(f"Inference time for frame {frame_number}/{total_frames}: {inference_time:.4f} seconds")

            # Draw bounding box on the frame
            cv.rectangle(frame, (state[0], state[1]), (state[0] + state[2], state[1] + state[3]), (0, 255, 0), 2)

            # Write the frame to the output video file
            if output_video is not None:
                output.write(frame)

        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Overall FPS: {total_frames / total_time:.2f}")

        # Release resources
        cap.release()
        if output_video is not None:
            output.release()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(input_video).stem
            base_results_path = os.path.join(self.results_dir, f"video_{video_name}")

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = f"{base_results_path}.txt"
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")

    def export2onnx(self, input_video, init_bbox):
        """Run the tracker with the videofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        params.debug = getattr(params, 'debug', 0)

        # Create tracker
        tracker = self.create_tracker(params)

        # Read video and get first frame
        cap = cv.VideoCapture(input_video)
        success, frame = cap.read()

        # Initialize tracker
        init_bbox = {'init_bbox': init_bbox}
        tracker.initialize(frame, init_bbox)

        # Get model we want to export
        model = tracker.network

        # Get dummy input on the next frame by using track method
        ret, frame = cap.read()
        frame_disp = frame.copy()
        template, appearance_features, search, seq_input = tracker.preprocess_input(frame)

        with torch.no_grad():
            device = 'cpu'
            template = template.to(device).type(torch.FloatTensor)
            appearance_features = appearance_features.to(device).type(torch.FloatTensor)
            search = search.to(device).type(torch.FloatTensor)
            seq_input = seq_input.to(device).type(torch.FloatTensor)
            dummy_input = (template, appearance_features, search, seq_input)
            print(f"{template.shape}")
            print(f"{appearance_features.shape}")
            print(f"{search.shape}")
            print(f"{seq_input.shape}")
            # print(f"{dummy_input=}")
            onnx_path = "tracking.onnx"
            input_names = ['template', 'appearance_features', 'search', 'seq_input']
            output_names = ['predicted_tokens', 'sequence_scores', 'sequence_features', 'score', 'refined_appearance_features']
            print('\n Exporting................... \n')
            torch.onnx.export(model=model, args=dummy_input, f=onnx_path, verbose=True, input_names=input_names, output_names=output_names, opset_version=15)

    def run_onnx(self, input_video, init_bbox, input_onnx):
        """Run the tracker with the video file and sum the inference time for each frame.
        args:
            input_video
            init_bbox
            input_onnx
        """
        
        params = self.get_parameters()

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        params.debug = getattr(params, 'debug', 0)

        # Create two trackers for comparing results between pytorch model and onnx model
        tracker_test = self.create_tracker(params)
        tracker_onnx = self.create_tracker(params)

        # Read video and get first frame
        cap = cv.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError("Unable to open video file")

        # Initialize trackers
        init_bbox = {'init_bbox': init_bbox}
        ret, frame = cap.read()  # Read the first frame
        if not ret:
            raise ValueError("Unable to read the first frame from the video")
        
        tracker_test.initialize(frame, init_bbox)
        tracker_onnx.initialize(frame, init_bbox)

        # Get pre-processed input for the ONNX model
        template, appearance_features, search, seq_input = tracker_onnx.preprocess_input(frame)

        # Check the model (optional)
        onnx_model = onnx.load(input_onnx)
        onnx.checker.check_model(onnx_model)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # Try CUDA first, fall back to CPU if necessary
        ort_session = onnxruntime.InferenceSession(input_onnx, providers=providers)

        total_time_ort = 0.0  # Initialize total time accumulator
        total_time_original = 0.0  # Initialize total time accumulator
        frame_count = 0  # Initialize frame count
        
        while True:
            ret, frame = cap.read()  # Read the next frame
            if not ret:
                break  # Exit the loop when the video ends

            # Prepare inputs for ONNX inference
            ort_inputs = {
                'template': template.detach().cpu().numpy(),
                'appearance_features': appearance_features.detach().cpu().numpy(),
                'search': search.detach().cpu().numpy(),
                'seq_input': seq_input.detach().cpu().numpy()
            }

            tic = time.time()  # Start timing inference
            ort_outputs = ort_session.run(None, ort_inputs)
            inference_time_ort = time.time() - tic  # Calculate the time taken for this frame

            tic = time.time()

            with torch.no_grad():
                original_outputs = tracker_test.network.forward(template, appearance_features, search, seq_input)
            inference_time_original = time.time() - tic  # Calculate the time taken for this frame
            
            total_time_ort += inference_time_ort  # Accumulate the inference time
            total_time_original += inference_time_original
            frame_count += 1  # Increment the frame count

            # Optionally, print results for this frame (for debugging)
            print(f"Frame {frame_count}: Onnx inference took {inference_time_ort:.4f} seconds vs Original inference took {inference_time_original:.4f} seconds")
            print(f"Test Output Seqs: {original_outputs['seqs']} vs ONNX Output: {ort_outputs[0]}")
        
        # After processing all frames, print the total time
        print(f"Total time for {frame_count} frames: Onnx: {total_time_ort:.4f} seconds vs Original: {total_time_original:.4f}")
        print(f"Average inference time per frame: Onnx: {total_time_ort / frame_count:.4f} seconds vs Original : {total_time_original / frame_count:.4f}")
