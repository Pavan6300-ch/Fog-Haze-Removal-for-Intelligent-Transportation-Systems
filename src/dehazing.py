
import cv2
import numpy as np

def dark_channel(image, window_size=15):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (window_size, window_size)
    )
    return cv2.erode(min_channel, kernel)


def estimate_atmospheric_light_v2(image, dark):
    h, w = dark.shape
    num_pixels = max(int(h * w * 0.001), 1)

    flat_dark = dark.ravel()
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]

    candidates = image.reshape(-1, 3)[indices]
    brightness = np.mean(candidates, axis=1)

    top_count = max(int(len(candidates) * 0.1), 1)
    top_indices = np.argsort(brightness)[-top_count:]

    return np.mean(candidates[top_indices], axis=0).reshape(1, 1, 3)


def guided_filter_transmission(image, transmission, radius=80, eps=0.04):
    guide = cv2.cvtColor(
        (image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    ).astype(np.float32) / 255.0

    refined = cv2.ximgproc.guidedFilter(
        guide=guide,
        src=transmission.astype(np.float32),
        radius=radius,
        eps=eps
    )

    return np.clip(refined, 0, 1)


def recover_scene(image, transmission, atmospheric_light, t0=0.20):
    transmission = np.maximum(transmission, t0)

    recovered = (
        (image - atmospheric_light) /
        transmission[:, :, np.newaxis]
    ) + atmospheric_light

    return np.clip(recovered, 0, 1)


def dehaze_image(image):
    """
    Remove haze from an RGB image using Dark Channel Prior.
    Input: uint8 RGB image.
    Output: uint8 RGB dehazed image.
    """

    image = image.astype(np.float32) / 255.0

    dark = dark_channel(image, window_size=15)

    atmospheric_light = estimate_atmospheric_light_v2(image, dark)

    transmission = 1 - 0.65 * dark / (
        np.mean(atmospheric_light) + 1e-6
    )

    transmission = np.clip(transmission, 0, 1)

    transmission = guided_filter_transmission(
        image, transmission, radius=80, eps=0.04
    )

    recovered = recover_scene(
        image, transmission, atmospheric_light, t0=0.20
    )

    return (recovered * 255).astype(np.uint8)
