import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import corruptions as C  # noqa: E402


def sample_images(n=8):
    """Synthetic sign-like 32x32 BGR images (no dataset needed): coloured disc, white inner ring, dark glyph."""
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[:32, :32]
    out = []
    for i in range(n):
        img = np.full((32, 32, 3), rng.randint(60, 140, 3), np.uint8)
        r = np.hypot(yy - 16, xx - 16)
        img[r < 13] = (30, 30, 200 - 10 * i)
        img[r < 9] = (240, 240, 240)
        img[12:20, 10 + i % 4:22 - i % 3] = (20, 20, 20)
        img = cv2.GaussianBlur(img, (0, 0), 0.8)                       # soft edges not aligned to any pixel grid
        tex = cv2.GaussianBlur(rng.randn(32, 32).astype(np.float32), (0, 0), 1.2) * 40
        out.append(np.clip(img + tex[..., None], 0, 255).astype(np.uint8))
    return out


IMGS = sample_images()


@pytest.mark.parametrize("name", C.NAMES)
def test_shape_dtype_preserved(name):
    for s in range(1, 6):
        out = C.FUNCS[name](IMGS[0], s, np.random.default_rng(0))
        assert out.shape == IMGS[0].shape and out.dtype == np.uint8


@pytest.mark.parametrize("name", C.NAMES)
def test_same_seed_same_output(name):
    a = C.FUNCS[name](IMGS[0], 3, np.random.default_rng(5))
    b = C.FUNCS[name](IMGS[0], 3, np.random.default_rng(5))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", C.NAMES)
def test_input_not_modified(name):
    img = IMGS[1].copy()
    ref = img.copy()
    for s in range(1, 6):
        C.FUNCS[name](img, s, np.random.default_rng(1))
    assert np.array_equal(img, ref)


def mean_diff(name, s):
    return np.mean([np.abs(C.FUNCS[name](im, s, np.random.default_rng(100 + i)).astype(float) - im).mean()
                    for i, im in enumerate(IMGS)])


@pytest.mark.parametrize("name", C.NAMES)
def test_difference_increases_with_severity(name):
    d = [mean_diff(name, s) for s in range(1, 6)]
    assert all(b > a for a, b in zip(d, d[1:])), d


@pytest.mark.parametrize("name", C.NAMES)
def test_severity1_changes_image(name):
    assert mean_diff(name, 1) > 0


def test_batch_deterministic_and_independent_of_jobs():
    X = np.stack(IMGS * 250)
    a = C.corrupt_batch(X, "gaussian_noise", 3, "t", n_jobs=1)
    b = C.corrupt_batch(X, "gaussian_noise", 3, "t", n_jobs=2)
    assert np.array_equal(a, b)
    assert np.array_equal(C.corrupt_batch(X[:5], "snow", 0, "t"), X[:5])
