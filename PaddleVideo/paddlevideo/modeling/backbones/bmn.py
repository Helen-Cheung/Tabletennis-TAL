# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import random
import paddle
import paddle.nn as nn
from paddle import ParamAttr
from ..registry import BACKBONES

class TemporalShift_random(nn.Layer):
    def __init__(self, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift_random, self).__init__()
        # self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.channels_range = list(range(400))  # feature_channels
        if inplace:
            print('=> Using in-place shift...')
        # print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        # self.fold_div = n_div
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace, channels_range =self.channels_range)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div=8, inplace=False, channels_range=[1,2]):
        x = paddle.transpose(x, perm=[0, 2, 1])   # [B,C,T] --> [B, T, C]
        x = x.numpy()
        # set_trace()
        n_batch, T, c = x.shape[0], x.shape[1], x.shape[2]
        # nt, c, h, w = x.size()
        # n_batch = nt // n_segment
        # x = x.view(n_batch, n_segment, c, h, w)
        # x = x.view(n_batch, T, c, h, w)
        fold = c // fold_div
        all = random.sample(channels_range, fold*2)
        forward = sorted(all[:fold])
        backward = sorted(all[fold:])
        fixed = list(set(channels_range) - set(all))
        # fold = c // fold_div

        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            # out = torch.zeros_like(x)
            # out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            # out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            # out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
            out = np.zeros_like(x)
            out[:, :-1, forward] = x[:, 1:, forward]  # shift left
            out[:, 1:, backward] = x[:, :-1, backward]  # shift right
            out[:, :, fixed] = x[:, :, fixed]  # not shift

        # return out.view(nt, c, h, w)
        out = paddle.to_tensor(out)
        out = paddle.transpose(out, perm=[0, 2, 1])
        return out

def _get_interp1d_bin_mask(seg_xmin, seg_xmax, tscale, num_sample,
                           num_sample_perbin):
    """ generate sample mask for a boundary-matching pair """
    plen = float(seg_xmax - seg_xmin)
    plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
    total_samples = [
        seg_xmin + plen_sample * ii
        for ii in range(num_sample * num_sample_perbin)
    ]
    p_mask = []
    for idx in range(num_sample):
        bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) *
                                    num_sample_perbin]
        bin_vector = np.zeros([tscale])
        for sample in bin_samples:
            sample_upper = math.ceil(sample)
            sample_decimal, sample_down = math.modf(sample)
            if (tscale - 1) >= int(sample_down) >= 0:
                bin_vector[int(sample_down)] += 1 - sample_decimal
            if (tscale - 1) >= int(sample_upper) >= 0:
                bin_vector[int(sample_upper)] += sample_decimal
        bin_vector = 1.0 / num_sample_perbin * bin_vector
        p_mask.append(bin_vector)
    p_mask = np.stack(p_mask, axis=1)
    return p_mask


def get_interp1d_mask(tscale, dscale, prop_boundary_ratio, num_sample,
                      num_sample_perbin):
    """ generate sample mask for each point in Boundary-Matching Map """
    mask_mat = []
    for start_index in range(tscale):
        mask_mat_vector = []
        for duration_index in range(dscale):
            if start_index + duration_index < tscale:
                p_xmin = start_index
                p_xmax = start_index + duration_index
                center_len = float(p_xmax - p_xmin) + 1
                sample_xmin = p_xmin - center_len * prop_boundary_ratio
                sample_xmax = p_xmax + center_len * prop_boundary_ratio
                p_mask = _get_interp1d_bin_mask(sample_xmin, sample_xmax,
                                                tscale, num_sample,
                                                num_sample_perbin)
            else:
                p_mask = np.zeros([tscale, num_sample])
            mask_mat_vector.append(p_mask)
        mask_mat_vector = np.stack(mask_mat_vector, axis=2)
        mask_mat.append(mask_mat_vector)
    mask_mat = np.stack(mask_mat, axis=3)
    mask_mat = mask_mat.astype(np.float32)

    sample_mask = np.reshape(mask_mat, [tscale, -1])
    return sample_mask


def init_params(name, in_channels, kernel_size):
    fan_in = in_channels * kernel_size * 1
    k = 1. / math.sqrt(fan_in)
    param_attr = ParamAttr(name=name,
                           initializer=paddle.nn.initializer.Uniform(low=-k,
                                                                     high=k))
    return param_attr


@BACKBONES.register()
class BMN(paddle.nn.Layer):
    """BMN model from
    `"BMN: Boundary-Matching Network for Temporal Action Proposal Generation" <https://arxiv.org/abs/1907.09702>`_
    Args:
        tscale (int): sequence length, default 100.
        dscale (int): max duration length, default 100.
        prop_boundary_ratio (float): ratio of expanded temporal region in proposal boundary, default 0.5.
        num_sample (int): number of samples betweent starting boundary and ending boundary of each propoasl, default 32.
        num_sample_perbin (int):  number of selected points in each sample, default 3.
    """

    def __init__(
        self,
        tscale,
        dscale,
        prop_boundary_ratio,
        num_sample,
        num_sample_perbin,
        feat_dim=400,
        shift=False,
    ):
        super(BMN, self).__init__()

        #init config
        self.feat_dim = feat_dim
        self.tscale = tscale
        self.dscale = dscale
        self.prop_boundary_ratio = prop_boundary_ratio
        self.num_sample = num_sample
        self.num_sample_perbin = num_sample_perbin
        self.shift = shift

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        # Base Module
        self.b_conv1 = paddle.nn.Conv1D(
            in_channels=self.feat_dim,
            out_channels=self.hidden_dim_1d,
            kernel_size=3,
            padding=1,
            groups=4,
            weight_attr=init_params('Base_1_w', self.feat_dim, 3),
            bias_attr=init_params('Base_1_b', self.feat_dim, 3))
        self.b_conv1_act = paddle.nn.ReLU()

        self.b_conv2 = paddle.nn.Conv1D(
            in_channels=self.hidden_dim_1d,
            out_channels=self.hidden_dim_1d,
            kernel_size=3,
            padding=1,
            groups=4,
            weight_attr=init_params('Base_2_w', self.hidden_dim_1d, 3),
            bias_attr=init_params('Base_2_b', self.hidden_dim_1d, 3))
        self.b_conv2_act = paddle.nn.ReLU()

        # Temporal Evaluation Module
        self.t_conv1 = paddle.nn.Conv1D(
            in_channels=self.hidden_dim_1d,
            out_channels=self.hidden_dim_1d,
            kernel_size=3,
            padding=1,
            groups=4,
            weight_attr=init_params('TEM_t1_w', self.hidden_dim_1d, 3),
            bias_attr=init_params('TEM_t1_b', self.hidden_dim_1d, 3))
        self.t_conv1_act = paddle.nn.ReLU()

        self.ts_conv1 = paddle.nn.Conv1D(
            in_channels=self.hidden_dim_1d,
            out_channels=self.hidden_dim_2d,
            kernel_size=3,
            padding=1,
            groups=1,
            weight_attr=init_params('TEM_s1_w', self.hidden_dim_1d, 1),
            bias_attr=init_params('TEM_s1_b', self.hidden_dim_1d, 1))
        self.ts_conv1_act = paddle.nn.ReLU()

        self.ts_conv2 = paddle.nn.Conv1D(
            in_channels=self.hidden_dim_2d,
            out_channels=self.hidden_dim_2d,
            kernel_size=3,
            padding=1,
            groups=1,
            weight_attr=init_params('TEM_s2_w', self.hidden_dim_2d, 1),
            bias_attr=init_params('TEM_s2_b', self.hidden_dim_2d, 1))
        self.ts_conv2_act = paddle.nn.ReLU()

        self.te_conv1 = paddle.nn.Conv1D(
            in_channels=self.hidden_dim_1d,
            out_channels=self.hidden_dim_2d,
            kernel_size=3,
            padding=1,
            groups=1,
            weight_attr=init_params('TEM_e1_w', self.hidden_dim_1d, 1),
            bias_attr=init_params('TEM_e1_b', self.hidden_dim_1d, 1))
        self.te_conv1_act = paddle.nn.ReLU()

        self.te_conv2 = paddle.nn.Conv1D(
            in_channels=self.hidden_dim_2d,
            out_channels=self.hidden_dim_2d,
            kernel_size=3,
            padding=1,
            groups=1,
            weight_attr=init_params('TEM_e2_w', self.hidden_dim_2d, 1),
            bias_attr=init_params('TEM_e2_b', self.hidden_dim_2d, 1))
        self.te_conv2_act = paddle.nn.ReLU()

        self.t_conv2 = paddle.nn.Conv1D(
            in_channels=self.hidden_dim_1d,
            out_channels=2,
            kernel_size=1,
            padding=0,
            groups=1,
            weight_attr=init_params('TEM_t2_w', self.hidden_dim_1d, 1),
            bias_attr=init_params('TEM_t2_b', self.hidden_dim_1d, 1))
        self.t_conv2_act = paddle.nn.Sigmoid()

        #Proposal Evaluation Module
        self.p_conv1 = paddle.nn.Conv1D(
            in_channels=self.hidden_dim_1d,
            out_channels=self.hidden_dim_2d,
            kernel_size=3,
            padding=1,
            groups=1,
            weight_attr=init_params('PEM_1d_w', self.hidden_dim_1d, 3),
            bias_attr=init_params('PEM_1d_b', self.hidden_dim_1d, 3))
        self.p_conv1_act = paddle.nn.ReLU()

        # init to speed up
        sample_mask = get_interp1d_mask(self.tscale, self.dscale,
                                        self.prop_boundary_ratio,
                                        self.num_sample, self.num_sample_perbin)
        self.sample_mask = paddle.to_tensor(sample_mask)
        self.sample_mask.stop_gradient = True

        self.p_conv3d1 = paddle.nn.Conv3D(
            in_channels=128,
            out_channels=self.hidden_dim_3d,
            kernel_size=(self.num_sample, 1, 1),
            stride=(self.num_sample, 1, 1),
            padding=0,
            weight_attr=ParamAttr(name="PEM_3d1_w"),
            bias_attr=ParamAttr(name="PEM_3d1_b"))
        self.p_conv3d1_act = paddle.nn.ReLU()

        self.p_conv2d1 = paddle.nn.Conv2D(
            in_channels=512,
            out_channels=self.hidden_dim_2d,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name="PEM_2d1_w"),
            bias_attr=ParamAttr(name="PEM_2d1_b"))
        self.p_conv2d1_act = paddle.nn.ReLU()

        self.p_conv2d2 = paddle.nn.Conv2D(
            in_channels=128,
            out_channels=self.hidden_dim_2d,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(name="PEM_2d2_w"),
            bias_attr=ParamAttr(name="PEM_2d2_b"))
        self.p_conv2d2_act = paddle.nn.ReLU()

        self.p_conv2d3 = paddle.nn.Conv2D(
            in_channels=128,
            out_channels=self.hidden_dim_2d,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(name="PEM_2d3_w"),
            bias_attr=ParamAttr(name="PEM_2d3_b"))
        self.p_conv2d3_act = paddle.nn.ReLU()

        self.p_conv2d4 = paddle.nn.Conv2D(
            in_channels=128,
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name="PEM_2d4_w"),
            bias_attr=ParamAttr(name="PEM_2d4_b"))
        self.p_conv2d4_act = paddle.nn.Sigmoid()

    def init_weights(self):
        pass

    def forward(self, x):
        if self.shift:
            temporal_perb = TemporalShift_random(400, 64)   
            x = temporal_perb(x)

        #Base Module
        x = self.b_conv1(x)
        x = self.b_conv1_act(x)
        x = self.b_conv2(x)
        x = self.b_conv2_act(x)

        #TEM
        xt = self.t_conv1(x)

        xs = self.ts_conv1(xt)
        xs = self.ts_conv1_act(xs)
        xs = self.ts_conv2(xs)
        xs = self.ts_conv2_act(xs)

        xe = self.te_conv1(xt)
        xe = self.te_conv1_act(xe)
        xe = self.te_conv2(xe)
        xe = self.te_conv2_act(xe)

        xt = paddle.concat(x=[xs,xe],axis=1)
        xt = self.t_conv2(xt)
        xt = self.t_conv2_act(xt)
        xs = xt[:,0]
        xs = paddle.squeeze(xs, axis=[1])
        xe = xt[:,1]
        xe = paddle.squeeze(xe, axis=[1])

        #PEM
        xp = self.p_conv1(x)
        xp = self.p_conv1_act(xp)
        #BM layer
        xp = paddle.matmul(xp, self.sample_mask)
        xp = paddle.reshape(xp, shape=[0, 0, -1, self.dscale, self.tscale])

        xp = self.p_conv3d1(xp)
        xp = self.p_conv3d1_act(xp)
        xp = paddle.squeeze(xp, axis=[2])
        xp = self.p_conv2d1(xp)
        xp = self.p_conv2d1_act(xp)
        xp = self.p_conv2d2(xp)
        xp = self.p_conv2d2_act(xp)
        xp = self.p_conv2d3(xp)
        xp = self.p_conv2d3_act(xp)
        xp = self.p_conv2d4(xp)
        xp = self.p_conv2d4_act(xp)
        return xp, xs, xe
