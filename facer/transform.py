from typing import List, Dict, Callable, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import functools
from skimage import transform


class GetCropAndResizeMatrix:
    """GetCropAndResizeMatrix

        box (x1 y1 x2 y2) -> transform_matrix
    """

    def __init__(self, target_shape, target_face_scale=1.0, make_rec_square=True,
                 offset_xy=None, align_corners=True,
                 offset_box_coords=False):
        """
        Args:
            align_corners (bool): Set this to `True` only if the box you give has coordinates
                ranging from `0` to `h-1` or `w-1`.

            offset_box_coords (bool): Set this to `True` if the box you give has coordinates
                ranging from `0` to `h` or `w`. 

                Set this to `False` if the box coordinates range from `-0.5` to `h-0.5` or `w-0.5`.

                If the box coordinates range from `0` to `h-1` or `w-1`, set `align_corners=True`.
        """
        self.target_shape = target_shape
        self.target_face_scale = target_face_scale
        self.make_rec_square = make_rec_square
        if offset_xy is None:
            offset_xy = (0, 0)
        self.offset_xy = offset_xy
        self.align_corners = align_corners
        self.offset_box_coords = offset_box_coords

    def __call__(self, box):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2 + self.offset_xy[0]
        cy = (y1 + y2) / 2 + self.offset_xy[1]
        rx = (x2 - x1) / 2 / self.target_face_scale
        ry = (y2 - y1) / 2 / self.target_face_scale
        if self.make_rec_square:
            rx, ry = max(rx, ry), max(rx, ry)

        x1, y1, x2, y2 = cx - rx, cy - ry, cx + rx, cy + ry

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


class GetFaceAlignMatrix:
    """ multi_face_align_points -> tranform_matrices
        or
        single_face_align_points -> transform_matrix
    """

    def __init__(self, target_shape, target_face_scale=1.0,
                 offset_xy=None, target_pts=None):
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
        return matrix


@functools.lru_cache(maxsize=128)
def _meshgrid(h, w):
    yy, xx = np.meshgrid(np.arange(0, h, dtype=np.float32),
                         np.arange(0, w, dtype=np.float32),
                         indexing='ij')
    return yy, xx


def _forge_transform_map(output_shape, fn: Callable[[np.ndarray], np.ndarray]
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """ Forge transform maps with a given function `fn`.

    Args:
        output_shape (tuple): (h, w, ...).
        fn (Callable[[np.ndarray], np.ndarray]): The function that accepts 
            a Nx2 array and outputs the transformed Nx2 array. Both input 
            and output store (x, y) coordinates.

    Note: 
        both input and output arrays of `fn` should store (y, x) coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two maps `X` and `Y`, where for each pixel (y, x) or coordinate (x, y),
            `(X[y, x], Y[y, x]) = fn([x, y])`
    """
    h, w, *_ = output_shape
    yy, xx = _meshgrid(h, w)  # h x w
    in_xxyy = np.stack([xx, yy], axis=-1).reshape([-1, 2])  # (h x w) x 2
    out_xxyy: np.ndarray = fn(in_xxyy)  # (h x w) x 2
    return out_xxyy.reshape([h, w, 2])


def _safe_arctanh(x):
    x[x < -0.999] = -0.999
    x[x > +0.999] = +0.999
    x = np.arctanh(x)
    return x


def _tanh_warp_transform(coords, transform_matrix, warp_factor, warped_shape):
    """ Tanh-warp function.

    Args:
        coords (np.ndarray): N x 2 (x, y).
        transform_matrix: 3 x 3. A matrix that transforms un-normalized coordinates 
            from the original image to the aligned yet not-warped image.
        warp_factor (float): The warp factor. 
            0 means linear transform, 1 means full tanh warp.
        warped_shape (tuple): [height, width].

    Returns:
        np.ndarray: N x 2 (x, y).
    """
    h, w, *_ = warped_shape
    # h -= 1
    # w -= 1

    if warp_factor > 0:
        # normalize coordinates to [-1, +1]
        coords = coords / np.array([w, h], dtype=coords.dtype) * 2 - 1

        nl_part1 = coords > 1.0 - warp_factor
        nl_part2 = coords < -1.0 + warp_factor

        coords[nl_part1] = _safe_arctanh(
            (coords[nl_part1] - 1.0 + warp_factor) /
            warp_factor) * warp_factor + \
            1.0 - warp_factor
        coords[nl_part2] = _safe_arctanh(
            (coords[nl_part2] + 1.0 - warp_factor) /
            warp_factor) * warp_factor - \
            1.0 + warp_factor

        # denormalize
        coords = (coords + 1) / 2 * np.array([w, h], dtype=coords.dtype)

    coords_homo = np.concatenate(
        [coords, np.ones([coords.shape[0], 1], dtype=coords.dtype)], axis=1)  # N x 3

    inv_matrix = np.linalg.inv(transform_matrix)
    coords_homo = np.dot(coords_homo, inv_matrix.T)  # N x 3
    return (coords_homo[:, :2] / coords_homo[:, [2, 2]]).astype(coords.dtype)


def _inverted_tanh_warp_transform(coords, transform_matrix, warp_factor, warped_shape):
    """ Inverted Tanh-warp function.

    Args:
        coords (np.ndarray): N x 2 (x, y).
        transform_matrix: 3 x 3. A matrix that transforms un-normalized coordinates 
            from the original image to the aligned yet not-warped image.
        warp_factor (float): The warp factor. 
            0 means linear transform, 1 means full tanh warp.
        warped_shape (tuple): [height, width].

    Returns:
        np.ndarray: N x 2 (x, y).
    """
    h, w, *_ = warped_shape
    # h -= 1
    # w -= 1

    coords_homo = np.concatenate(
        [coords, np.ones([coords.shape[0], 1], dtype=coords.dtype)], axis=1)  # N x 3

    coords_homo = np.dot(coords_homo, transform_matrix.T)  # N x 3
    coords = (coords_homo[:, :2] / coords_homo[:, [2, 2]]).astype(coords.dtype)

    if warp_factor > 0:
        # normalize coordinates to [-1, +1]
        coords = coords / np.array([w, h], dtype=coords.dtype) * 2 - 1

        nl_part1 = coords > 1.0 - warp_factor
        nl_part2 = coords < -1.0 + warp_factor

        coords[nl_part1] = np.tanh(
            (coords[nl_part1] - 1.0 + warp_factor) /
            warp_factor) * warp_factor + \
            1.0 - warp_factor
        coords[nl_part2] = np.tanh(
            (coords[nl_part2] + 1.0 - warp_factor) /
            warp_factor) * warp_factor - \
            1.0 + warp_factor

        # denormalize
        coords = (coords + 1) / 2 * np.array([w, h], dtype=coords.dtype)

    return coords


class GetTransformMap:
    """ transform_matrix -> map

    This involves an implementation of the Tanh-warping, which is proposed in:

        [1] Lin, Jinpeng, Hao Yang, Dong Chen, Ming Zeng, Fang Wen, and Lu Yuan. 
            "Face parsing with roi tanh-warping." In Proceedings of the IEEE/CVF 
            Conference on Computer Vision and Pattern Recognition, pp. 5654-5663. 2019.

        [2] Zheng, Yinglin, Hao Yang, Ting Zhang, Jianmin Bao, Dongdong Chen, Yangyu Huang, 
            Lu Yuan, Dong Chen, Ming Zeng, and Fang Wen. "General Facial Representation 
            Learning in a Visual-Linguistic Manner." arXiv preprint arXiv:2112.03109 (2021).

    Please cite the paper on your usage.

    """

    def __init__(self, warped_shape, warp_factor=0.0):
        """
        Args:
            warped_shape: The target image shape to transform to.
            warp_factor: The warping factor. `warp_factor=1.0` represents a vannila Tanh-warping, 
                `warp_factor=0.0` represents a cropping.
        """
        self.warped_shape = warped_shape
        self.warp_factor = warp_factor

    def __call__(self, matrix):
        return _forge_transform_map(
            self.warped_shape,
            functools.partial(_tanh_warp_transform,
                              transform_matrix=matrix,
                              warp_factor=self.warp_factor,
                              warped_shape=self.warped_shape))


class GetInvertedTransformMap:
    """ transform_matrix, original_image_shape -> inverted_map

    This is an inverted transform of GetTransformMap.

    This involves an implementation of the Tanh-warping, which is proposed in:

        [1] Lin, Jinpeng, Hao Yang, Dong Chen, Ming Zeng, Fang Wen, and Lu Yuan. 
            "Face parsing with roi tanh-warping." In Proceedings of the IEEE/CVF 
            Conference on Computer Vision and Pattern Recognition, pp. 5654-5663. 2019.

        [2] Zheng, Yinglin, Hao Yang, Ting Zhang, Jianmin Bao, Dongdong Chen, Yangyu Huang, 
            Lu Yuan, Dong Chen, Ming Zeng, and Fang Wen. "General Facial Representation 
            Learning in a Visual-Linguistic Manner." arXiv preprint arXiv:2112.03109 (2021).

    Please cite the paper on your usage.

    """

    def __init__(self, warped_shape, warp_factor=0.0):
        """
        Args:
            warped_shape: The target image shape to transform to.
            warp_factor: The warping factor. `warp_factor=1.0` represents a vannila Tanh-warping, 
                `warp_factor=0.0` represents a cropping.
        """
        self.warped_shape = warped_shape
        self.warp_factor = warp_factor

    def __call__(self, data):
        matrix, orig_shape = data
        return _forge_transform_map(
            orig_shape,
            functools.partial(_inverted_tanh_warp_transform,
                              transform_matrix=matrix,
                              warp_factor=self.warp_factor,
                              warped_shape=self.warped_shape))


def get_grid(images: torch.Tensor, data: List[Dict[str, torch.Tensor]],
             get_matrix, src_name: str, target_shape, warp_factor: float = 0.0
             ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    """
    returns image_ids, align_grid, inv_align_grid
    """

    get_transform_map = GetTransformMap(target_shape, warp_factor=warp_factor)
    get_inv_transform_map = GetInvertedTransformMap(
        target_shape, warp_factor=warp_factor)

    assert len(data) == images.size(0)
    target_h, target_w = target_shape[:2]
    origin_h, origin_w = images.shape[2:]

    transform_maps = []
    inv_transform_maps = []
    image_ids = []

    for image_id, datum in enumerate(data):
        batch_src = datum[src_name].cpu().numpy()  # points (n x 5 x 2) or rect (n x 4)
        for src in batch_src:
            matrix = get_matrix(src)  # 4 x 4

            transform_map = get_transform_map(matrix)  # h x w x 2
            transform_maps.append(transform_map)

            inv_transform_map = get_inv_transform_map(
                (matrix, (origin_h, origin_w)))
            inv_transform_maps.append(inv_transform_map)

            image_ids.append(image_id)

    transform_maps = np.stack(transform_maps, axis=0)  # (b*n) x h x w x 2
    # (b*n) x origin_h x origin_w x 2
    inv_transform_maps = np.stack(inv_transform_maps, axis=0)

    # make it satisfy F.grid_sample

    transform_maps[:, :, :, 0] = (
        transform_maps[:, :, :, 0] / origin_w) * 2 - 1
    transform_maps[:, :, :, 1] = (
        transform_maps[:, :, :, 1] / origin_h) * 2 - 1
    transform_maps = torch.from_numpy(transform_maps).to(images.device)

    inv_transform_maps[:, :, :, 0] = (
        inv_transform_maps[:, :, :, 0] / target_w) * 2 - 1
    inv_transform_maps[:, :, :, 1] = (
        inv_transform_maps[:, :, :, 1] / target_h) * 2 - 1
    inv_transform_maps = torch.from_numpy(inv_transform_maps).to(images.device)

    return image_ids, transform_maps, inv_transform_maps


def get_face_align_grid(images: torch.Tensor, data: List[Dict[str, torch.Tensor]],
                        target_shape, target_face_scale=1.0,
                        offset_xy=None, target_pts=None, warp_factor: float = 0.0
                        ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    """
    returns image_ids, align_grid, inv_align_grid
    """
    get_matrix = GetFaceAlignMatrix(
        target_shape, target_face_scale=target_face_scale,
        offset_xy=offset_xy, target_pts=target_pts)
    return get_grid(images, data, get_matrix, 'points', target_shape, warp_factor)


def get_face_crop_grid(images: torch.Tensor, data: List[Dict[str, torch.Tensor]],
                       target_shape, target_face_scale=1.0,
                       offset_xy=None, warp_factor: float = 0.0
                       ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    """
    returns image_ids, align_grid, inv_align_grid
    """
    get_matrix = GetCropAndResizeMatrix(
        target_shape, target_face_scale=target_face_scale,
        offset_xy=offset_xy)
    return get_grid(images, data, get_matrix, 'rects', target_shape, warp_factor)

