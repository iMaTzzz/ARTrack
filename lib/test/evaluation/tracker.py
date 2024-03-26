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

    def __init__(self, name: str, parameter_name: str, run_id: int = None, display_name: str = None):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)

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

    def run_video(self, input_video, output_video, bbox_path=None, debug=None, visdom_info=None, save_results=False):
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
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(input_video), "Invalid param {}".format(input_video)
        ", input_video must be a valid videofile"

        output_boxes = []

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

        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(input_video))
            exit(-1)
        if bbox_path is not None:
            assert isinstance(bbox_path, (list, tuple))
            assert len(bbox_path) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(bbox_path))
            output_boxes.append(bbox_path)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break
            if output_video is not None:
                output.write(frame_disp)
        cv.destroyAllWindows()

        frame_number = 1
        total_time = 0
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            frame_number += 1
            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)
            inference_time = out['time']
            total_time += inference_time
            print(f"Inference time for frame {frame_number}/{total_frames}: {inference_time:.4f} seconds")

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            # font_color = (0, 0, 0)
            # cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       # font_color, 1)
            # cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       # font_color, 1)
            # cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       # font_color, 1)

            # # Display the resulting frame
            # cv.imshow(display_name, frame_disp)
            # key = cv.waitKey(1)
            # if key == ord('q'):
                # break
            # elif key == ord('r'):
                # ret, frame = cap.read()
                # frame_disp = frame.copy()

                # cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           # (0, 0, 0), 1)

                # cv.imshow(display_name, frame_disp)
                # x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                # init_state = [x, y, w, h]
                # tracker.initialize(frame, _build_init_info(init_state))
                # output_boxes.append(init_state)
            # Save frame with predictions
            if output_video is not None:
                output.write(frame_disp)
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Overall FPS: {total_frames / total_time:.2f}")

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(input_video).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
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
        model = tracker.get_network()

        # Get dummy input on the next frame by using track method
        ret, frame = cap.read()
        frame_disp = frame.copy()
        template, search, seq_input = tracker.preprocess_input(frame_disp)

        with torch.no_grad():
            device = 'cpu'
            template = template.to(device).type(torch.FloatTensor)
            search = search.to(device).type(torch.FloatTensor)
            seq_input = seq_input.to(device).type(torch.FloatTensor)
            dummy_input = (template, search, seq_input)
            print(f"{template=}, {template.shape}")
            print(f"{search=}, {search.shape}")
            print(f"{seq_input=}, {seq_input.shape}")
            print(f"{dummy_input=}")
            onnx_path = "tracking.onnx"
            input_names = ['template', 'search', 'seq_input']
            output_names = ['seqs', 'class', 'feat', 'x_feat', 'backbone_feat']
            print('\n Exporting................... \n')
            torch.onnx.export(model=model, args=dummy_input, f=onnx_path, verbose=True, input_names=input_names, output_names=output_names, opset_version=11)

    def run_onnx(self, input_video, init_bbox, input_onnx):
        """Run the tracker with the videofile.
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
        success, frame = cap.read()

        # Initialize trackers
        init_bbox = {'init_bbox': init_bbox}
        tracker_test.initialize(frame, init_bbox)
        tracker_onnx.initialize(frame, init_bbox)

        # Get the next frame
        ret, frame = cap.read()
        frame_copy = frame.copy()
        frame_copy2 = frame.copy()

        # Track the get the output result to compare
        tracker_test.track(frame_copy)

        # Get pre-processed input
        template, search, seq_input = tracker_onnx.preprocess_input(frame_copy2)
        # print("Pre-processed inputs:")
        # print(f"{template=}")
        # print(f"{search=}")
        # print(f"{seq_input=}")
        
        # Check the model
        onnx_model = onnx.load(input_onnx)
        onnx.checker.check_model(onnx_model)
        # print('Model :\n\n{}'.format(onnx.helper.printable_graph(onnx_model.graph)))

        ort_session = onnxruntime.InferenceSession(input_onnx)
        ort_inputs = {'template': template.cpu().numpy(), 'search': search.cpu().numpy(), 'seq_input': seq_input.cpu().numpy()}
        ort_ouputs = ort_session.run(None, ort_inputs)
        print(f"{ort_ouputs=}")
        # print(f"{out['seqs']=} \n {out_seqs}")
        # print(f"{out['class']=} \n {out_class}")
        # print(f"{out['feat']=} \n {out_feat}")
        # print(f"{out['x_feat']=} \n {out_x_feat}")
        # print(f"{out['backbone_feat']=} \n {out_backbone_feat}")