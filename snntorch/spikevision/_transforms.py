# Adapted from https://github.com/nmi-lab/torchneuromorphic by Emre Neftci and Clemens Schaefer

import numpy as np
import pandas as pd
import torch, bisect
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda


def find_first(a, tgt):
    return bisect.bisect_left(a, tgt)


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


class toOneHot(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, integers):
        y_onehot = torch.FloatTensor(integers.shape[0], self.num_classes)
        y_onehot.zero_()
        return y_onehot.scatter_(1, torch.LongTensor(integers), 1)


class toDtype(object):
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, integers):
        return torch.tensor(integers, dtype=self.dtype)


class Downsample(object):
    """Resize the address event Tensor to the given size.
    Args:
        factor: : Desired resize factor. Applied to all dimensions including time
    """

    def __init__(self, factor):
        assert isinstance(factor, int) or hasattr(factor, "__iter__")
        self.factor = factor

    def __call__(self, tmad):
        return tmad // self.factor

    def __repr__(self):
        return self.__class__.__name__ + "(dt = {0}, dp = {1}, dx = {2}, dy = {3})"


# class Crop(object):
#    def __init__(self, low_crop, high_crop):
#        '''
#        Crop all dimensions
#        '''
#        self.low = low_crop
#        self.high = high_crop
#
#    def __call__(self, tmad):
#        idx = np.where(np.any(tmad>high_crop, axis=1))
#        tmad = np.delete(tmad,idx,0)
#        idx = np.where(np.any(tmad<high_crop, axis=1))
#        tmad = np.delete(tmad,idx,0)
#        return tmad
#
#    def __repr__(self):
#        return self.__class__.__name__ + '()'

# class ShuffleMask(object):
#     '''
#     Shuffles events within t_min and t_max with a Random spike train having the same number of events
#     '''
#     def __init__(self, t_min, t_max, size=[2,32,32]):
#         from decolle.snn_utils import spiketrains
#         self.generator = spiketrains
#         self.t_min = t_min
#         self.t_max = t_max
#         self.size = size

#     def __call__(self, tmad):
#         idx = np.where((tmad[:,0]>=self.t_min) * (tmad[:,0]<self.t_max))
#         for i in range(1,tmad.shape[1]):
#             #tmad[idx,i] = shuffle_along_axis(tmad[idx,i][0],0)
#             tmad[idx,i] = np.random.randint(low=0,high=self.size[i-1],size=len(idx[0]))

#         return tmad

#     def __repr__(self):
#         return self.__class__.__name__ + '()'


class CropDims(object):
    def __init__(self, low_crop, high_crop, dims):
        self.low_crop = low_crop
        self.high_crop = high_crop
        self.dims = dims

    def __call__(self, tmad):
        for i, d in enumerate(self.dims):
            idx = np.where(tmad[:, d] >= self.high_crop[i])
            tmad = np.delete(tmad, idx, 0)
            idx = np.where(tmad[:, d] < self.low_crop[i])
            tmad = np.delete(tmad, idx, 0)
            # Normalize
            tmad[:, d] = tmad[:, d] - self.low_crop[i]
        return tmad

    def __repr__(self):
        return self.__class__.__name__ + "()"


class CropCenter(object):
    def __init__(self, center, size):
        self.center = np.array(center, dtype=np.uint32)
        self.att_shape = np.array(size[1:], dtype=np.uint32)
        self.translation = (center - self.att_shape // 2).astype(np.uint32)

    def __call__(self, tmad):
        trans = np.repeat(self.translation[np.newaxis, :], len(tmad), axis=0)
        tmad[:, 2:] -= trans
        idx = np.where(np.any(tmad[:, 2:] >= self.att_shape, axis=1))
        tmad = np.delete(tmad, idx, 0)
        idx = np.where(np.any(tmad[:, 2:] < [0, 0], axis=1))
        tmad = np.delete(tmad, idx, 0)
        return tmad

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Attention(object):
    def __init__(self, n_attention_events, size):
        """
        Crop around the median event in the last n_events.
        """
        self.att_shape = np.array(size[1:], dtype=np.int64)
        self.n_att_events = n_attention_events

    def __call__(self, tmad):
        df = pd.DataFrame(tmad, columns=["t", "p", "x", "y"])
        # compute centroid in x and y
        centroids = (
            df.loc[:, ["x", "y"]]
            .rolling(window=self.n_att_events, min_periods=1)
            .median()
            .astype(int)
        )
        # re-address (translate) events with respect to centroid corner
        df.loc[:, ["x", "y"]] -= centroids - self.att_shape // 2
        # remove out of range events
        df = df.loc[
            (df.x >= 0)
            & (df.x < self.att_shape[1])
            & (df.y >= 0)
            & (df.y < self.att_shape[0])
        ]
        return df.to_numpy()

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ToChannelHeightWidth(object):
    def __call__(self, tmad):
        n = tmad.shape[1]
        if n == 2:
            o = np.zeros(tmad.shape[0], dtype=tmad.dtype)
            return np.column_stack([tmad, o, o])

        elif n == 4:
            return tmad

        else:
            raise TypeError(
                "Wrong number of dimensions. Found {0}, expected 1 or 3".format(n - 1)
            )

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ToCountFrame(object):
    """Convert Address Events to Binary tensor.
    Converts a numpy.ndarray (T x H x W x C) to a torch.FloatTensor of shape (T x C x H x W) in the range [0., 1., ...]
    """

    def __init__(self, T=500, size=[2, 32, 32]):
        self.T = T
        self.size = size
        self.ndim = len(size)

    def __call__(self, tmad):
        times = tmad[:, 0]
        t_start = times[0]
        t_end = times[-1]
        addrs = tmad[:, 1:]

        ts = range(0, self.T)
        chunks = np.zeros([len(ts)] + self.size, dtype="int8")
        idx_start = 0
        idx_end = 0
        for i, t in enumerate(ts):
            idx_end += find_first(times[idx_end:], t + 1)
            if idx_end > idx_start:
                ee = addrs[idx_start:idx_end]
                i_pol_x_y = tuple([i] + [ee[:, j] for j in range(self.ndim)])
                np.add.at(chunks, i_pol_x_y, 1)
            idx_start = idx_end
        return chunks

    def __repr__(self):
        return self.__class__.__name__ + "(T={0})".format(self.T)


class ToEventSum(object):
    """Convert Address Events to Image By Summing.
    Converts a numpy.ndarray (T x H x W x C) to a torch.FloatTensor of shape (C x H x W) in the range [0., 1., ...]
    """

    def __init__(self, T=500, size=[2, 32, 32]):
        self.T = T
        self.size = size
        self.ndim = len(size)

    def __call__(self, tmad):
        times = tmad[:, 0]
        t_start = times[0]
        t_end = times[-1]
        addrs = tmad[:, 1:]

        ts = range(0, self.T)
        chunks = np.zeros([len(ts)] + self.size, dtype="int8")
        idx_start = 0
        idx_end = 0
        for i, t in enumerate(ts):
            idx_end += find_first(times[idx_end:], t)
            if idx_end > idx_start:
                ee = addrs[idx_start:idx_end]
                i_pol_x_y = tuple([i] + [ee[:, j] for j in range(self.ndim)])
                np.add.at(chunks, i_pol_x_y, 1)
            idx_start = idx_end
        return chunks.sum(axis=0, keepdims=True)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class FilterEvents(object):
    def __init__(self, kernel=None, groups=1, tpad=None):
        self.kernel = kernel
        self.groups = groups
        if tpad is None:
            self.tpad = self.kernel.shape[2] // 2
        else:
            self.tpad = tpad

    def __call__(self, chunks):
        chunks = chunks.to(self.kernel.device)
        if len(chunks.shape) == 4:
            data = chunks.permute([1, 0, 2, 3])
            data = data.unsqueeze(0)
        else:
            data = chunks.permute([0, 2, 1, 3, 4])

        Y = torch.nn.functional.conv3d(
            data, self.kernel, groups=self.groups, padding=[self.tpad, 0, 0]
        )
        Y = Y.transpose(1, 2)
        if len(chunks.shape) == 4:
            return Y[0]
        else:
            return Y


class ExpFilterEvents(FilterEvents):
    def __init__(self, length, tau=200, channels=2, tpad=None, device="cpu", **kwargs):
        t = torch.arange(0.0, length, 1.0)
        kernel = torch.ones(channels, 1, len(t), 1, 1)
        exp_kernel = torch.exp(-t / tau)
        exp_kernel /= exp_kernel.sum()
        exp_kernel = torch.flip(
            exp_kernel, [0]
        )  # Conv3d is cross correlation not convolution
        groups = 2

        for i in range(channels):
            kernel[i, 0, :, 0, 0] = exp_kernel
        kernel = kernel.to(device)

        super(ExpFilterEvents, self).__init__(kernel, groups, tpad, **kwargs)


class Rescale(object):
    """Rescale the event sum Tensor by the given factor.
    Args:
        factor: : Desired rescale factor.
    """

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, chunks):
        return chunks * self.factor

    def __repr__(self):
        return self.__class__.__name__ + "({0})".format(self.factor)


# debug
class hflip(object):
    """Horizontally flip the given image.

    Args:
        img (PIL Image or Tensor): Image to be flipped. If img
            is a Tensor, it is expected to be in [..., H, W] format,
            where ... means it can have an arbitrary number of leading
            dimensions.

    Returns:
        PIL Image or Tensor:  Horizontally flipped image.
    """

    def __call__(self, target):
        return target.flip([-1])

    def __repr__(self):
        return self.__class__.__name__ + "()"


class rot90(object):
    """Rotate given image 90 degrees CW."""

    def __call__(self, target):
        return target.rot90(1, [-2, -1])

    def __repr__(self):
        return self.__class__.__name__ + "()"


class dvs_permute(object):
    def __call__(self, target):
        return target.flip([-2]).rot90(1, [-1, -2])

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Repeat(object):
    """
    Replicate np.array (C) as (n_repeat X C). This is useful to transform sample labels into sequences
    """

    def __init__(self, n_repeat):
        self.n_repeat = n_repeat

    def __call__(self, target):
        return np.tile(np.expand_dims(target, 0), [self.n_repeat, 1])

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray to a torch.FloatTensor of the same shape
    """

    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, frame):
        """
        Args:
            frame (numpy.ndarray): numpy array of frames
        Returns:
            Tensor: Converted data.
        """
        return torch.FloatTensor(frame).to(self.device)

    def __repr__(self):
        return self.__class__.__name__ + "(device:{0})".format(self.device)
