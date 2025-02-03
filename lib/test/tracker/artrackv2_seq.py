import math

from lib.models.artrackv2_seq import build_artrackv2_seq
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import time


class ARTrackV2Seq(BaseTracker):
    def __init__(self, params):
        super(ARTrackV2Seq, self).__init__(params)
        network = build_artrackv2_seq(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)

        self.cfg = params.cfg
        self.bins = params.cfg.MODEL.BINS
        self.network = network
        # self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.dz_feat = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True)
        # self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.frame_id = 0
        # for save boxes from all queries
        self.store_result = None
        self.prenum = params.cfg.MODEL.PRENUM
        self.range = params.cfg.MODEL.RANGE
        self.x_feat = None

    def initialize(self, image, info: dict):
        # forward the template once
        self.x_feat = None
        self.update_ = False

        z_patch_arr, resize_factor, _ = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)  # output_sz=self.params.template_size
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        with torch.no_grad():
            self.template = template
            self.dz_feat = self.network.backbone.patch_embed(template)

        self.box_mask_z = None

        # save states
        self.state = info['init_bbox']
        self.store_result = [info['init_bbox'].copy()]
        for i in range(self.prenum - 1):
            self.store_result.append(info['init_bbox'].copy())
        self.frame_id = 0

    def track(self, image):
        # Time initialization
        tic = time.time()
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, _ = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        for i in range(len(self.store_result)):
            box_temp = self.store_result[i].copy()
            box_out_i = transform_image_to_crop(torch.Tensor(self.store_result[i]), torch.Tensor(self.state),
                                                resize_factor,
                                                torch.Tensor([self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE]),
                                                normalize=True)
            box_out_i[2] = box_out_i[2] + box_out_i[0]
            box_out_i[3] = box_out_i[3] + box_out_i[1]
            box_out_i = box_out_i.clamp(min=-0.5, max=1.5)
            box_out_i = (box_out_i + 0.5) * (self.bins - 1)
            if i == 0:
                seqs_out = box_out_i
            else:
                seqs_out = torch.cat((seqs_out, box_out_i), dim=-1)

        seqs_out = seqs_out.unsqueeze(0)

        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            # merge the template and the search
            # run the transformer
            # template = torch.concat([self.z_dict1.tensors.unsqueeze(1), self.z_dict2.tensors.unsqueeze(1)], dim=1)
            out_dict = self.network.forward(
                template=self.template, dz_feat=self.dz_feat, search=search, seq_input=seqs_out)

        self.dz_feat = out_dict['dz_feat']
        self.x_feat = out_dict['x_feat']

        pred_boxes = (out_dict['seqs'][:, 0:4] + 0.5) / (self.bins - 1) - 0.5

        pred_feat = out_dict['feat']
        pred = pred_feat.permute(1, 0, 2).reshape(-1, self.bins * self.range + 6)

        pred = pred_feat[0:4, :, 0:self.bins * self.range]

        out = pred.softmax(-1).to(pred)
        mul = torch.range((-1 * self.range * 0.5 + 0.5) + 1 / (self.bins * self.range), (self.range * 0.5 + 0.5) - 1 / (self.bins * self.range), 2 / (self.bins * self.range)).to(pred)

        ans = out * mul
        ans = ans.sum(dim=-1)
        ans = ans.permute(1, 0).to(pred)

        pred_boxes = (ans + pred_boxes) / 2

        pred_boxes = pred_boxes.view(-1, 4).mean(dim=0)

        pred_new = pred_boxes
        pred_new[2] = pred_boxes[2] - pred_boxes[0]
        pred_new[3] = pred_boxes[3] - pred_boxes[1]
        pred_new[0] = pred_boxes[0] + pred_new[2] / 2
        pred_new[1] = pred_boxes[1] + pred_new[3] / 2

        pred_boxes = (pred_new * self.params.search_size / resize_factor).tolist()

        self.state = clip_box(self.map_box_back(pred_boxes, resize_factor), H, W, margin=10)

        if len(self.store_result) < self.prenum:
            self.store_result.append(self.state.copy())
        else:
            for i in range(self.prenum):
                if i != self.prenum - 1:
                    self.store_result[i] = self.store_result[i + 1]
                else:
                    self.store_result[i] = self.state.copy()

        out = dict()

        out["target_bbox"] = self.state
        # Record time taken per inference
        out['time'] = time.time() - tic
        return out

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        # cx_real = cx + cx_prev
        # cy_real = cy + cy_prev
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights

    def preprocess_input(self, image):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, _ = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        if self.dz_feat == None:
            self.dz_feat = self.network.backbone.patch_embed(self.z_dict2.tensors)
        for i in range(len(self.store_result)):
            box_temp = self.store_result[i].copy()
            box_out_i = transform_image_to_crop(torch.Tensor(self.store_result[i]), torch.Tensor(self.state),
                                                resize_factor,
                                                torch.Tensor([self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE]),
                                                normalize=True)
            box_out_i[2] = box_out_i[2] + box_out_i[0]
            box_out_i[3] = box_out_i[3] + box_out_i[1]
            box_out_i = box_out_i.clamp(min=-0.5, max=1.5)
            box_out_i = (box_out_i + 0.5) * (self.bins - 1)
            if i == 0:
                seqs_out = box_out_i
            else:
                seqs_out = torch.cat((seqs_out, box_out_i), dim=-1)

        seqs_out = seqs_out.unsqueeze(0)

        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            template = torch.concat([self.z_dict1.tensors.unsqueeze(1), self.z_dict2.tensors.unsqueeze(1)], dim=1)
        print(f"template={template}, dz_feat={self.dz_feat}, x_dict.tensors={x_dict.tensors}, seqs_out={seqs_out}")
        return template, self.dz_feat, x_dict.tensors, seqs_out


def get_tracker_class():
    return ARTrackV2Seq
