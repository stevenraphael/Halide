#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "HalideBuffer.h"
#include "HalideRuntime.h"

#include "../support/mem_probe.h"
#include "stereobm_inductive.h"
#include "stereobm_noninductive.h"
#include "stereobm_unfolded.h"

#include "../support/bench_harness.h"
#include "halide_image_io.h"

#if STEREOBM_BUILD_OPENCV
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

using namespace Halide::Tools;

// winsize/depth are baked into the AOT libraries (GeneratorParams) and passed as
// -D by build_all.sh; keep defaults so the driver also builds standalone. They
// are needed here to report the analytic sliding-window storage fold.
#ifndef WINSIZE
static constexpr int WINSIZE = 9;
#endif
#ifndef DEPTH
static constexpr int DEPTH = 16;
#endif

#if STEREOBM_BUILD_OPENCV
#ifndef PREFILTER_CAP
static constexpr int PREFILTER_CAP = 31;
#endif
#ifndef TEXTURE_THRESHOLD
static constexpr int TEXTURE_THRESHOLD = 10;
#endif
#ifndef UNIQUENESS_RATIO
static constexpr int UNIQUENESS_RATIO = 0;
#endif
#ifndef MIN_DISP
static constexpr int MIN_DISP = 0;
#endif
#endif

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Usage: %s left right out\n", argv[0]);
        return 1;
    }

    // Default 1: this is a small image, so single-threaded is both the most
    // stable and the fastest configuration for both backends.
    int num_threads = 1;
    if (const char *nt = getenv("NUM_THREADS")) {
        num_threads = std::max(1, atoi(nt));
    }
    halide_set_num_threads(num_threads);
#if STEREOBM_BUILD_OPENCV
    cv::setNumThreads(num_threads);
#endif
    printf("threads: %d\n", num_threads);

    Halide::Runtime::Buffer<uint8_t, 2> left, right;

    // Load + grayscale via Halide's image IO for BOTH paths. This keeps the
    // OpenCV comparison free of OpenCV's image-codec modules (imgcodecs and its
    // png/jpeg/tiff/webp tree), so the binary needs only core/imgproc/calib3d --
    // and it means OpenCV's StereoBM runs on byte-identical input to ours.
    auto load_gray = [](const char *path) {
        Halide::Runtime::Buffer<uint8_t, 3> color = load_and_convert_image(path);
        Halide::Runtime::Buffer<uint8_t, 2> gray(color.width(), color.height());
        gray.for_each_element([&](int i, int j) {
            uint32_t r = color(i, j, 0), g = color(i, j, 1), b = color(i, j, 2);
            gray(i, j) = (uint8_t)((r * 4899 + g * 9617 + b * 1868 + 8192) >> 14);
        });
        return gray;
    };
    left = load_gray(argv[1]);
    right = load_gray(argv[2]);

#if STEREOBM_BUILD_OPENCV
    // Wrap the same dense grayscale buffers as cv::Mat (no copy) for StereoBM.
    cv::Mat cv_left(left.height(), left.width(), CV_8UC1, left.data());
    cv::Mat cv_right(right.height(), right.width(), CV_8UC1, right.data());
#endif

    Halide::Runtime::Buffer<int16_t, 2> output(left.width(), left.height());
    Halide::Runtime::Buffer<int16_t, 2> output_uf(left.width(), left.height());
    Halide::Runtime::Buffer<int16_t, 2> output_ni(left.width(), left.height());

    hb::Stats s_ind = hb::bench([&]() {
        stereobm_inductive(left, right, output);
        output.device_sync();
    });
    // Inductive UNFOLDED: same fusion, cost-volume storage pinned to full extent.
    hb::Stats s_unf = hb::bench([&]() {
        stereobm_unfolded(left, right, output_uf);
        output_uf.device_sync();
    });
    hb::Stats s_non = hb::bench([&]() {
        stereobm_noninductive(left, right, output_ni);
        output_ni.device_sync();
    });

    // Measured peak internal scratch (untimed, separate from the benches above).
    double bytes_ind = hb::measure_aot_peak([&]() { stereobm_inductive(left, right, output); });
    double bytes_unf = hb::measure_aot_peak([&]() { stereobm_unfolded(left, right, output_uf); });
    double bytes_non = hb::measure_aot_peak([&]() { stereobm_noninductive(left, right, output_ni); });

    // Correctness gate: inductive (folded & unfolded) must match non-inductive
    // bit-for-bit.
    long long mismatches = 0;
    int max_abs_diff = 0;
    output.for_each_element([&](int i, int j) {
        int diff = std::abs((int)output(i, j) - (int)output_ni(i, j));
        if (diff != 0) {
            mismatches++;
            max_abs_diff = std::max(max_abs_diff, diff);
        }
    });
    long long mismatches_uf = 0;
    output_uf.for_each_element([&](int i, int j) {
        if (output_uf(i, j) != output_ni(i, j)) mismatches_uf++;
    });
    const bool ind_ok = (mismatches == 0 && mismatches_uf == 0);

    // Mean abs disparity error (px) of each Halide backend vs OpenCV, the
    // third-party oracle. Left <0 (UNCHECKED) when OpenCV isn't compiled in.
    double err_ind = -1.0, err_non = -1.0;
    bool opencv_ok = true;

#if STEREOBM_BUILD_OPENCV
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(DEPTH, WINSIZE);
    bm->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
    bm->setPreFilterCap(PREFILTER_CAP);
    bm->setMinDisparity(MIN_DISP);
    bm->setTextureThreshold(TEXTURE_THRESHOLD);
    bm->setUniquenessRatio(UNIQUENESS_RATIO);

    cv::Mat cv_disp;
    hb::Stats s_cv = hb::bench([&]() {
        bm->compute(cv_left, cv_right, cv_disp);
    });

    const int32_t invalid = (MIN_DISP - 1) * 16;
    if (output.width() != cv_disp.cols || output.height() != cv_disp.rows) {
        fprintf(stderr, "Output size %d x %d doesn't match OpenCV's output size %d x %d\n",
                output.width(), output.height(), cv_disp.cols, cv_disp.rows);
        return 1;
    }
    auto mean_abs_vs_cv = [&](Halide::Runtime::Buffer<int16_t, 2> &buf) {
        long long compared = 0, sum_abs = 0;
        buf.for_each_element([&](int i, int j) {
            int32_t h = buf(i, j), o = cv_disp.at<short>(j, i);
            if (h == invalid || o == invalid) return;
            compared++;
            sum_abs += std::abs(h - o);
        });
        return compared > 0 ? (sum_abs / (double)compared) / 16.0 : -1.0;
    };
    err_ind = mean_abs_vs_cv(output);
    err_non = mean_abs_vs_cv(output_ni);
    opencv_ok = (err_ind >= 0 && err_ind < 2.0);  // within 2px of OpenCV
#endif
    double err_unf = err_ind;  // folded and unfolded are bit-identical when ind_ok

    // The sliding-window SAD keeps only O(winsize) rows of the depth x W cost
    // volume in the inductive form; the non-inductive form materializes all H
    // rows. That H/winsize fold is the standardized state-footprint story.
    const double mpix = (double)left.width() * left.height() / 1e6;
    char note[160];
    snprintf(note, sizeof(note),
             "Stereo block matching  %dx%d  winsize=%d depth=%d tilesize baked  |  peak-scratch fold %.1fx",
             left.width(), left.height(), WINSIZE, DEPTH, hb::mem_ratio(bytes_non, bytes_ind));
    hb::print_spec_header("stereobm", "host", note);
#if STEREOBM_BUILD_OPENCV
    hb::print_row("OpenCV StereoBM (third-party)", s_cv, mpix / (s_cv.min * 1e-3),
                  "Mpix/s", 0.0, 0.0, true);
#endif
    hb::print_row("non-inductive (materialize)", s_non, mpix / (s_non.min * 1e-3),
                  "Mpix/s", bytes_non, err_non, opencv_ok);
    hb::print_row("inductive UNFOLDED (fold rows -> H)", s_unf, mpix / (s_unf.min * 1e-3),
                  "Mpix/s", bytes_unf, err_unf, ind_ok && opencv_ok,
                  hb::verdict(s_unf.min, s_non.min));
    hb::print_row("inductive FOLDED (fold rows -> 1)", s_ind, mpix / (s_ind.min * 1e-3),
                  "Mpix/s", bytes_ind, err_ind, ind_ok && opencv_ok,
                  hb::verdict(s_ind.min, s_unf.min));
    printf("  inductive vs non-inductive: %lld / %d mismatches (max abs diff %d); unfolded mismatches %lld\n",
           mismatches, output.width() * output.height(), max_abs_diff, mismatches_uf);

    int32_t dmin = INT32_MAX, dmax = INT32_MIN;
    output.for_each_value([&](int16_t v) {
        dmin = std::min(dmin, (int32_t)v);
        dmax = std::max(dmax, (int32_t)v);
    });
    double scale = (dmax > dmin) ? 255.0 / (dmax - dmin) : 0.0;
    Halide::Runtime::Buffer<uint8_t, 2> vis(output.width(), output.height());
    vis.for_each_element([&](int i, int j) {
        int n = (output(i, j) - dmin) * scale + 0.5;
        vis(i, j) = std::max(0, std::min(255, n));
    });
    convert_and_save_image(vis, argv[3]);

    const bool pass = ind_ok && opencv_ok;
    printf("%s\n", pass ? "Success!" : "FAILED (correctness gate)");
    return pass ? 0 : 1;
}
