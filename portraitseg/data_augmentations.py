import numpy as np


def apply_data_aug(inp, target, data_aug, kwargs=None):
    """
    Applies the transform to the input and target in the same way.
    """
    concatenation = np.concatenate([inp, target[None, :]])
    if kwargs:
        concatenation_aug = data_aug(concatenation, **kwargs)
    else:
        concatenation_aug = data_aug(concatenation)
    inp, target = concatenation_aug[:-1], concatenation_aug[-1]
    return inp, target


def mirror(inp, target):
    """
    - Horizontal flip
    - Appropriate for image transformation tasks
    - Requires the width axis to be the last axis
    """
    def mirror(image):
        return np.flip(image, -1).copy()
    inp, target = apply_data_aug(inp, target, mirror)
    return inp, target


def random_crop(inp, target, crop_percent=0.875):
    def random_crop(x, crop_percent=0.875, sync_seed=None):
        # https://github.com/fchollet/keras/issues/3338
        np.random.seed(sync_seed)
        h, w = x.shape[1:] if x.ndim == 3 else x.shape
        crop_h, crop_w = int(h * crop_percent), int(w * crop_percent)
        range_h, range_w = (h - crop_h) // 2, (w - crop_w) // 2
        offset_h = 0 if range_h == 0 else np.random.randint(range_h)
        offset_w = 0 if range_w == 0 else np.random.randint(range_w)
        if x.ndim == 3:
            return x[:, offset_h:offset_h+crop_h, offset_w:offset_w+crop_w]
        elif x.ndim == 2:
            return x[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w]
        else:
            raise Exception("Invalid dimensions")
    inp, target = apply_data_aug(inp, target, random_crop,
                                 {"crop_percent": crop_percent})
    return inp, target
