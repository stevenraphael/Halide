"""Validate + time the Halide integral-image box filter against OpenCV."""
import time
import numpy as np
import cv2

with open("apps/integral/params.txt") as f:
    W, H, R = map(int, f.readline().split())
inp = np.fromfile("apps/integral/in.bin", dtype=np.float32).reshape(H, W)
hal = np.fromfile("apps/integral/box.bin", dtype=np.float64).reshape(H, W)

k = 2 * R + 1
# Window SUM (normalize=False), replicate border to match the Halide clamp.
cv = cv2.boxFilter(inp.astype(np.float64), ddepth=cv2.CV_64F, ksize=(k, k), normalize=False,
                   borderType=cv2.BORDER_REPLICATE)

# Compare interior (>= R from each edge) where border handling is irrelevant.
s = slice(R, H - R), slice(R, W - R)
num = np.abs(hal[s] - cv[s])
den = np.abs(cv[s]) + 1e-6
rel = float(np.max(num / den))
print(f"Integral box filter vs OpenCV boxFilter  W={W} H={H} R={R} (window {k}x{k})")
print(f"  max rel error (interior) = {rel:.3e} -> {'PASS' if rel < 1e-4 else 'FAIL'}")

inp64 = inp.astype(np.float64)
def bench(fn, n=10):
    best = 1e18
    for _ in range(n):
        t0 = time.perf_counter(); fn(); best = min(best, (time.perf_counter()-t0)*1e3)
    return best
t_box = bench(lambda: cv2.boxFilter(inp64, ddepth=cv2.CV_64F, ksize=(k, k),
                                    normalize=False, borderType=cv2.BORDER_REPLICATE))
t_int = bench(lambda: cv2.integral(inp64))
print(f"  OpenCV boxFilter (separable running sum): {t_box:.3f} ms")
print(f"  OpenCV integral() (materialize SAT only): {t_int:.3f} ms")
