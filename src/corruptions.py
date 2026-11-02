import zlib

import cv2
import numpy as np
from scipy import ndimage

# Each corruption maps to its family, five severity values and a short description of the value
PARAMS = {
    "gaussian_noise": ("noise", [0.04, 0.08, 0.12, 0.18, 0.26], "sigma of additive N(0,sigma) on [0,1] pixels"),
    "shot_noise": ("noise", [300, 120, 50, 20, 8], "Poisson lambda (photons per unit intensity; smaller = noisier)"),
    "impulse_noise": ("noise", [0.02, 0.05, 0.09, 0.14, 0.20], "fraction of salt-and-pepper pixel values"),
    "defocus_blur": ("blur", [0.8, 1.1, 1.5, 2.0, 2.5], "disk kernel radius in px"),
    "motion_blur": ("blur", [3, 4, 5, 7, 9], "line kernel length in px (random angle)"),
    "zoom_blur": ("blur", [1.08, 1.16, 1.26, 1.38, 1.52], "max centre zoom factor averaged over 8 zoom steps"),
    "snow": ("weather", [0.02, 0.04, 0.06, 0.09, 0.12], "fraction of pixels that start a flake"),
    "frost": ("weather", [0.2, 0.3, 0.4, 0.5, 0.6], "blend weight of the procedural frost texture"),
    "fog": ("weather", [0.3, 0.45, 0.6, 0.75, 0.9], "fog intensity (plasma-fractal haze opacity scale)"),
    "brightness": ("digital", [0.1, 0.2, 0.3, 0.4, 0.5], "constant added to HSV value channel"),
    "contrast": ("digital", [0.8, 0.6, 0.45, 0.3, 0.2], "scale of the deviation from the per-channel mean"),
    "pixelate": ("digital", [0.8, 0.65, 0.55, 0.4, 0.3], "downscale factor (box filter) then nearest upscale; 0.5 and 0.75 avoided: sizes of exactly 16 and 24 px have a dip in error, which breaks monotonicity"),
    "elastic_transform": ("digital", [0.8, 1.2, 1.7, 2.2, 2.8], "max displacement in px of the smoothed random field"),
    "jpeg_compression": ("digital", [50, 30, 18, 10, 5], "JPEG quality"),
}
NAMES = list(PARAMS)
FAMILIES = {}
for _n, (_f, _, _) in PARAMS.items():
    FAMILIES.setdefault(_f, []).append(_n)
FAMILY_OF = {n: v[0] for n, v in PARAMS.items()}


# Look up the parameter of a corruption for a severity from 1 to 5
def _p(name, severity):
    if not 1 <= severity <= 5:
        raise ValueError("severity must be 1..5")
    return PARAMS[name][1][severity - 1]


# Convert a uint8 image to floats between 0 and 1
def _f(img):
    return img.astype(np.float32) / 255.0


# Convert floats between 0 and 1 back to a uint8 image
def _u8(x):
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


# Build a smooth line shaped kernel for motion blur
def _line_kernel(length, angle):
    k = int(2 * np.ceil(length / 2) + 1)
    c = k // 2
    t = np.linspace(-length / 2, length / 2, int(length * 8) + 1)
    xs, ys = c + t * np.cos(angle), c + t * np.sin(angle)
    ker = np.zeros((k, k), np.float32)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs - x0, ys - y0
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(ker, (np.clip(y0 + dy, 0, k - 1), np.clip(x0 + dx, 0, k - 1)), w)
    return ker / ker.sum()


# Build a round kernel with a soft edge for defocus blur
def _disk_kernel(radius):
    k = int(2 * np.ceil(radius) + 1)
    c = k // 2
    yy, xx = np.mgrid[:k, :k]
    ker = np.clip(radius + 0.5 - np.hypot(yy - c, xx - c), 0, 1).astype(np.float32)
    return ker / ker.sum()


# Make a random cloud pattern with the diamond square method, size must be a power of two
def _plasma(size, rng, decay=2.0):
    m = np.zeros((size, size), np.float64)
    step, wibble = size, 100.0
    while step > 1:
        half = step // 2
        ref = m[0:size:step, 0:size:step]
        acc = ref + np.roll(ref, -1, 0)
        acc = acc + np.roll(acc, -1, 1)
        m[half:size:step, half:size:step] = acc / 4 + rng.uniform(-wibble, wibble, acc.shape)
        d = np.roll(m, half, 0) + np.roll(m, -half, 0) + np.roll(m, half, 1) + np.roll(m, -half, 1)
        for sl in (np.s_[0:size:step, half:size:step], np.s_[half:size:step, 0:size:step]):
            m[sl] = d[sl] / 4 + rng.uniform(-wibble, wibble, m[sl].shape)
        step, wibble = half, wibble / decay
    m -= m.min()
    return (m / (m.max() + 1e-9)).astype(np.float32)


# Resize a 2D array to the given height and width
def _to_size(a, h, w):
    return a if a.shape == (h, w) else cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)


# Make a cloud pattern that matches the image size
def _plasma_for(h, w, rng):
    size = 1 << int(np.ceil(np.log2(max(h, w))))
    return _to_size(_plasma(size, rng), h, w)


# Add Gaussian noise to every pixel
def gaussian_noise(img, severity, rng):
    x = _f(img)
    return _u8(x + rng.normal(0, _p("gaussian_noise", severity), x.shape).astype(np.float32))


# Simulate photon counting noise with a Poisson distribution
def shot_noise(img, severity, rng):
    lam = _p("shot_noise", severity)
    return _u8(rng.poisson(_f(img) * lam).astype(np.float32) / lam)


# Turn random pixel values fully black or fully white
def impulse_noise(img, severity, rng):
    a = _p("impulse_noise", severity)
    x = _f(img)
    u = rng.random(x.shape)
    x[u < a / 2] = 0.0
    x[(u >= a / 2) & (u < a)] = 1.0
    return _u8(x)


# Blur the image with a round kernel like an out of focus lens
def defocus_blur(img, severity, rng):
    return _u8(cv2.filter2D(_f(img), -1, _disk_kernel(_p("defocus_blur", severity)), borderType=cv2.BORDER_REFLECT))


# Blur the image along a line in a random direction
def motion_blur(img, severity, rng):
    ker = _line_kernel(_p("motion_blur", severity), rng.uniform(0, np.pi))
    return _u8(cv2.filter2D(_f(img), -1, ker, borderType=cv2.BORDER_REFLECT))


# Average several zoomed copies of the image around its centre
def zoom_blur(img, severity, rng):
    x = _f(img)
    h, w = x.shape[:2]
    acc = np.zeros_like(x)
    zooms = np.linspace(1.0, _p("zoom_blur", severity), 8)
    for z in zooms:
        M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), 0, float(z))
        acc += cv2.warpAffine(x, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return _u8(acc / len(zooms))


# Brighten the scene and draw falling snow streaks
def snow(img, severity, rng):
    d = _p("snow", severity)
    x = _f(img)
    h, w = x.shape[:2]
    flakes = (rng.random((h, w)) < d) * rng.uniform(0.6, 1.0, (h, w))
    flakes = flakes.astype(np.float32)
    ker = _line_kernel(rng.uniform(3, 5), np.pi / 2 + rng.uniform(-0.45, 0.45))
    layer = np.clip(cv2.filter2D(flakes, -1, ker * 2.5, borderType=cv2.BORDER_REFLECT), 0, 1)[..., None]
    gray = x.mean(axis=2, keepdims=True)
    bright = 0.1 + 1.5 * d
    x = (1 - bright) * x + bright * np.maximum(x, gray * 0.5 + 0.55)
    return _u8(1 - (1 - x) * (1 - layer))


# Blend a pale icy texture over the image
def frost(img, severity, rng):
    a = _p("frost", severity)
    x = _f(img)
    h, w = x.shape[:2]
    n = sum(wt * ndimage.gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), s, mode="wrap")
            for s, wt in ((0.7, 1.0), (1.5, 1.0), (3.0, 1.0)))
    n = n / (np.abs(n).max() + 1e-9)
    tex = 1 - np.abs(n)
    tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-9)
    tex = 0.35 + 0.65 * tex ** 2
    frost_img = tex[..., None] * np.array([1.0, 0.96, 0.86], np.float32)
    return _u8((1 - a) * x + a * frost_img)


# Cover the image with a patchy grey haze
def fog(img, severity, rng):
    f = _p("fog", severity)
    x = _f(img)
    h, w = x.shape[:2]
    m = _plasma_for(h, w, rng)[..., None]
    alpha = np.clip(0.8 * f * (0.45 + 0.55 * m), 0, 0.9)
    return _u8((1 - alpha) * x + alpha * 0.85)


# Add a constant to the brightness channel
def brightness(img, severity, rng):
    hsv = cv2.cvtColor(_f(img), cv2.COLOR_BGR2HSV)
    hsv[..., 2] = np.clip(hsv[..., 2] + _p("brightness", severity), 0, 1)
    return _u8(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))


# Squeeze the pixel values towards the mean of each channel
def contrast(img, severity, rng):
    x = _f(img)
    m = x.mean(axis=(0, 1), keepdims=True)
    return _u8((x - m) * _p("contrast", severity) + m)


# Shrink the image and scale it back up to make big blocks
def pixelate(img, severity, rng):
    h, w = img.shape[:2]
    nh, nw = max(2, int(round(h * _p("pixelate", severity)))), max(2, int(round(w * _p("pixelate", severity))))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# Warp the image with a smooth random displacement field
def elastic_transform(img, severity, rng):
    h, w = img.shape[:2]
    disp = []
    for _ in range(2):
        f = ndimage.gaussian_filter(rng.uniform(-1, 1, (h, w)).astype(np.float32), 2.5, mode="reflect")
        disp.append(f / (np.abs(f).max() + 1e-9) * _p("elastic_transform", severity))
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return cv2.remap(img, xx + disp[0], yy + disp[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


# Save and reload the image as a low quality JPEG
def jpeg_compression(img, severity, rng):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(_p("jpeg_compression", severity))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


FUNCS = {n: globals()[n] for n in NAMES}


# Corrupt one image, severity 0 returns an untouched copy
def apply(img, name, severity, rng):
    if severity == 0:
        return img.copy()
    return FUNCS[name](img, severity, rng)


# Fixed seed for a dataset, corruption and severity so results repeat
def seed_for(dataset, name, severity):
    return zlib.crc32(f"{dataset}|{name}|{severity}".encode()) & 0x7FFFFFFF


# Corrupt a chunk of images, each image gets its own generator from its index
def _chunk(images, idx, name, severity, base):
    import cv2 as _cv2
    _cv2.setNumThreads(0)
    return np.stack([apply(images[i], name, severity, np.random.default_rng([base, int(gi)])) for i, gi in enumerate(idx)])


# Corrupt many images, the result does not depend on the number of processes
def corrupt_batch(images, name, severity, dataset="", n_jobs=1):
    if severity == 0:
        return images.copy()
    base = seed_for(dataset, name, severity)
    if n_jobs == 1 or len(images) < 1500:
        return _chunk(images, range(len(images)), name, severity, base)
    from joblib import Parallel, delayed
    parts = np.array_split(np.arange(len(images)), n_jobs * 2)
    return np.concatenate(Parallel(n_jobs=n_jobs)(delayed(_chunk)(images[p], range(p[0], p[-1] + 1), name, severity, base) for p in parts))


# Save the corruption settings as a csv file
def write_params_csv(path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corruption", "family", "severity", "value", "parameter"])
        for n, (fam, vals, desc) in PARAMS.items():
            for s, v in enumerate(vals, 1):
                w.writerow([n, fam, s, v, desc])
