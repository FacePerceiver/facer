import torch
import numpy as np
import functools
from skimage import transform


class GetCropAndResizeMatrix:
    """GetCropAndResizeMatrix

    Args:
        align_corners (bool): Set this to `True` only if the box you give has coordinates
            ranging from `0` to `h-1` or `w-1`.

        offset_box_coords (bool): Set this to `True` if the box you give has coordinates
            ranging from `0` to `h` or `w`. 

            Set this to `False` if the box coordinates range from `-0.5` to `h-0.5` or `w-0.5`.

            If the box coordinates range from `0` to `h-1` or `w-1`, set `align_corners=True`.
    """

    def __init__(self, target_shape, align_corners, ret_shape=False,
                 offset_box_coords=False):
        self.target_shape = target_shape
        self.align_corners = align_corners
        self.ret_shape = ret_shape
        self.offset_box_coords = offset_box_coords

    def __call__(self, box):
        x1, y1, x2, y2 = box
        h, w, *_ = self.target_shape
        dtype = np.float32
        if self.align_corners:
            # x -> (x - x1) / (x2 - x1) * (w - 1)
            # y -> (y - y1) / (y2 - y1) * (h - 1)
            ax = 1.0 / (x2 - x1) * (w - 1)
            ay = 1.0 / (y2 - y1) * (h - 1)
            trans_matrix = np.array([
                [ax, 0, -x1 * ax],
                [0, ay, -y1 * ay],
                [0, 0, 1]
            ], dtype=dtype)
        else:
            if self.offset_box_coords:
                # x1, x2 \in [0, w], y1, y2 \in [0, h]
                # first we should offset x1, x2, y1, y2 to be ranging in
                # [-0.5, w-0.5] and [-0.5, h-0.5]
                # so to convert these pixel coordinates into boundary coordinates.
                x1, x2, y1, y2 = x1-0.5, x2-0.5, y1-0.5, y2-0.5

            # x -> (x - x1) / (x2 - x1) * w - 0.5
            # y -> (y - y1) / (y2 - y1) * h - 0.5
            ax = 1.0 / (x2 - x1) * w
            ay = 1.0 / (y2 - y1) * h
            trans_matrix = np.array([
                [ax, 0, -x1 * ax - 0.5],
                [0, ay, -y1 * ay - 0.5],
                [0, 0, 1]
            ], dtype=dtype)
        return trans_matrix


@functools.lru_cache()
def _standard_face_pts():
    pts = np.array([
        196.0, 226.0,
        316.0, 226.0,
        256.0, 286.0,
        220.0, 360.4,
        292.0, 360.4], np.float32) / 256.0 - 1.0
    return np.reshape(pts, (5, 2))


class GetFaceAlignMatrices:
    """ multi_face_align_points -> [out_shape], tranform_matrices
        or
        single_face_align_points -> [out_shape], transform_matrix
    """

    def __init__(self, target_shape, target_face_scale=1.0,
                 offset_xy=None, target_pts=None, ret_shape=False):
        if target_pts is None:
            std_pts = _standard_face_pts()  # [-1 1]
            h, w, *_ = target_shape
            self.target_pts = (std_pts * target_face_scale + 1) * \
                np.array([w-1, h-1], np.float32) / 2.0
            if offset_xy is not None:
                self.target_pts[:, 0] += offset_xy[0]
                self.target_pts[:, 1] += offset_xy[1]
        else:
            self.target_pts = np.array(target_pts)
        self.ret_shape = ret_shape

    def _estimate_single_face_align_matrix(self, fa_points):
        tform = transform.SimilarityTransform()
        tform.estimate(fa_points, self.target_pts)
        return tform.params

    def __call__(self, multi_fa_points):
        assert multi_fa_points.shape[-2:] == (5, 2)
        if multi_fa_points.ndim == 3:
            matrix = np.stack([self._estimate_single_face_align_matrix(fa_points)
                               for fa_points in multi_fa_points], 0)
        else:
            matrix = self._estimate_single_face_align_matrix(multi_fa_points)
        if self.ret_shape:
            return self.target_shape, matrix
        return matrix


GetFaceAlignMatrix = GetFaceAlignMatrices


def crop(images, data, tanh_warp: float = 0.0):
    pass
