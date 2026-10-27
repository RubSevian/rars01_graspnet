"""Restore Ultralytics masks from centered letterbox to original image pixels."""
import cv2
import numpy as np


def restore_mask(mask, image_shape):
    mask = np.asarray(mask, dtype=np.float32)
    h, w = image_shape
    if mask.ndim != 2 or min(mask.shape) <= 0 or h <= 0 or w <= 0 or not np.isfinite(mask).all():
        raise ValueError('Invalid segmentation mask dimensions or values')
    if mask.shape != (h, w):
        mh, mw = mask.shape
        gain = min(mh / h, mw / w)
        pad_w = (mw - round(w * gain)) / 2
        pad_h = (mh - round(h * gain)) / 2
        top, left = round(pad_h - .1), round(pad_w - .1)
        bottom, right = mh - round(pad_h + .1), mw - round(pad_w + .1)
        mask = mask[top:bottom, left:right]
        if not mask.size:
            raise ValueError('Empty segmentation after letterbox removal')
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > .5).astype(np.uint8)
