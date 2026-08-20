// Non-generator (JIT) StereoBM: the same block-matching stereo pipeline as
// stereobm_generator.cpp, but built and compiled in-process with compile_jit so
// it drops the slow AOT generator+.a build and reports through the standardized
// bench harness like the other inductive apps.
//
// Only ablation is the inductive column fold: vsum/textsum recur down y and
// clobber in place, so the folded window is a single row (fold_storage(y, 1));
// the unfolded variant pins them to the full image height, materializing the
// whole vertical trajectory while holding fusion fixed -- isolating folding from
// fusion exactly like viterbi/kalman/ode/chebyshev. Correctness is folded ==
// unfolded (identical math, different storage).
//
// Params match the generator defaults; override any via env (WINSIZE, DEPTH,
// TILESIZE, THRESHOLD, MINDISP, UNIQUENESS, FILTERCAP). Images from argv[1..2]
// (grayscale via Halide image IO); if absent, a synthetic shifted-gradient pair
// is generated so it runs standalone.
//
// Build: g++ apps/bench_standardized/stereobm_jit.cpp -O3 -march=native \
//   -std=c++17 -Ibuild/include -Iapps/support -Lbuild/src -lHalide \
//   -lpthread -ldl -o /tmp/stereobm_jit   (LD_LIBRARY_PATH=build/src)

#include "Halide.h"
#include "HalideBuffer.h"
#include "bench_harness.h"
#include "halide_image_io.h"
#include "mem_probe.h"
#include <cstdio>
#include <cstdlib>
#include <string>

#if STEREOBM_BUILD_OPENCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

using namespace Halide;

int main(int argc, char **argv) {
    // Params (compile-time constants so winsize/2 etc. fold).
    const int winsize = hb::env_int("WINSIZE", 9);
    const int depth = hb::env_int("DEPTH", 16);
    const int tilesize = hb::env_int("TILESIZE", 64);
    const int threshold = hb::env_int("THRESHOLD", 10);
    const int mindisp = hb::env_int("MINDISP", 0);
    const int uniqueness_ratio = hb::env_int("UNIQUENESS", 0);
    const int filtercap = hb::env_int("FILTERCAP", 31);
    // schedule4 (non-inductive, sliding-RDom) extra knobs. ytilesize defaults to
    // the (square) x tile so the tilesize sweep drives both; vector_width=0 =>
    // natural disparity vector width.
    const int ytilesize = hb::env_int("YTILESIZE", tilesize);
    const int vector_width = hb::env_int("VECTOR_WIDTH", 0);

    // The stereo pair is a REAL image (fixed size -- NOT a swept axis), so the
    // OpenCV timing comparison is meaningful. argv[1]/argv[2] = left/right paths.
    if (argc < 3) {
        printf("Usage: %s left.png right.png\n", argv[0]);
        return 1;
    }
    auto load_gray = [](const char *path) {
        Runtime::Buffer<uint8_t, 3> color = Tools::load_and_convert_image(path);
        Runtime::Buffer<uint8_t, 2> gray(color.width(), color.height());
        gray.for_each_element([&](int x, int y) {
            uint32_t r = color(x, y, 0), g = color(x, y, 1), b = color(x, y, 2);
            gray(x, y) = (uint8_t)((r * 4899 + g * 9617 + b * 1868 + 8192) >> 14);
        });
        return gray;
    };
    Runtime::Buffer<uint8_t, 2> lg = load_gray(argv[1]), rg = load_gray(argv[2]);
    int W = lg.width(), H = lg.height();
    // Wrap the loaded pixels as JIT Buffers for the pipeline.
    Buffer<uint8_t> left_gray(lg.data(), W, H), right_gray(rg.data(), W, H);

    // Match OpenCV's core count to Halide's for a fair timing comparison. Halide's
    // JIT thread count comes from HL_NUM_THREADS (set alongside this in run_tests).
    int num_threads = hb::env_int("STEREOBM_NUM_THREADS", hb::env_int("HL_NUM_THREADS", 1));

    const Type uint16 = UInt(16);
    const Type int16 = Int(16);
    const Type int32 = Int(32);
    const int native_lanes = get_jit_target_from_environment().natural_vector_size<int16_t>();

    auto build = [&](bool fold) -> Func {
        Var y("y"), x("x"), di("di"), xi("xi"), xo("xo");

        Func proc0("proc0"), proc1("proc1");
        proc0(x, y) = cast<int16_t>(BoundaryConditions::mirror_interior(left_gray)(x - winsize / 2, y - winsize / 2));
        proc1(x, y) = cast<int16_t>(BoundaryConditions::mirror_interior(right_gray)(x - winsize / 2, y - winsize / 2));

        Func xsobel0("xsobel0"), xsobel1("xsobel1");
        Expr e0 = proc0(x + 1, y - 1) - proc0(x - 1, y - 1) + 2 * proc0(x + 1, y) - 2 * proc0(x - 1, y) + proc0(x + 1, y + 1) - proc0(x - 1, y + 1);
        Expr e1 = proc1(x + 1, y - 1) - proc1(x - 1, y - 1) + 2 * proc1(x + 1, y) - 2 * proc1(x - 1, y) + proc1(x + 1, y + 1) - proc1(x - 1, y + 1);
        Expr ix = x - winsize / 2, iy = y - winsize / 2;
        Expr border = ix == 0 || ix == W - 1 || ((H % 2 == 1) && iy == H - 1);
        xsobel0(x, y) = select(border, cast<int16_t>(filtercap), cast<int16_t>(clamp(e0, -1 * filtercap, filtercap) + filtercap));
        xsobel1(x, y) = select(border, cast<int16_t>(filtercap), cast<int16_t>(clamp(e1, -1 * filtercap, filtercap) + filtercap));

        Func diff("diff");
        diff(di, xi, y, xo) = cast<uint16_t>(cast<uint16_t>(abs((xsobel0(xi + xo * tilesize, y)) - (xsobel1(xi + xo * tilesize + di - depth + 1 - mindisp, y)))));

        Func vsum(uint16, "vsum"), zero_blur(uint16, "zero_blur");
        RDom rwin(0, winsize, "rwin");
        zero_blur(di, xi, xo) = sum(cast<uint16_t>(diff(di, xi, rwin, xo)));
        vsum(di, xi, y, xo) = select(y <= 0, zero_blur(di, xi, xo), likely(vsum(di, xi, y - 1, xo) + diff(di, xi, y + winsize - 1, xo) - diff(di, xi, y - 1, xo)));
        Func blur_y(uint16, "blur_y");
        Func f1("f1");
        f1(di, y, xo) = sum(vsum(di, rwin, y, xo));
        blur_y(di, xi, y, xo) = select(xi <= 0, f1(di, y, xo), likely(blur_y(di, xi - 1, y, xo) + vsum(di, xi + winsize - 1, y, xo) - vsum(di, xi - 1, y, xo)));

        Func text("text"), zerotext("zerotext"), textf1("textf1");
        Func textsum(uint16, "textsum"), textblury(uint16, "textblury");
        text(xi, y, xo) = cast<uint8_t>(abs(cast<int16_t>(xsobel0(xi + xo * tilesize, y)) - cast<int16_t>(filtercap)));
        zerotext(xi, xo) = sum(cast<uint16_t>(text(xi, rwin, xo)));
        textsum(xi, y, xo) = select(y <= 0, zerotext(xi, xo), likely(textsum(xi, y - 1, xo) + text(xi, y + winsize - 1, xo) - text(xi, y - 1, xo)));
        textf1(y, xo) = sum(textsum(rwin, y, xo));
        textblury(xi, y, xo) = select(xi <= 0, textf1(y, xo), likely(textblury(xi - 1, y, xo) + textsum(xi + winsize - 1, y, xo) - textsum(xi - 1, y, xo)));

        Func preout("preout");
        RDom rd(0, depth, "rd");
        preout(xi, y, xo) = cast<uint16_t>(65535);
        preout(xi, y, xo) = min(blur_y(rd, xi, y, xo), preout(xi, y, xo));
        Func prearg("prearg");
        prearg(di, xi, y, xo) = select(preout(xi, y, xo) == blur_y(di, xi, y, xo), cast<uint16_t>(di), cast<uint16_t>(65535));
        Func argmin1("argmin1");
        argmin1(xi, y, xo) = cast<uint16_t>(65535);
        argmin1(xi, y, xo) = min(argmin1(xi, y, xo), prearg(rd, xi, y, xo));
        Func second_best("second_best");
        second_best(di, xi, y, xo) = select(abs(di - cast<int16_t>(argmin1(xi, y, xo))) <= 1, cast<uint16_t>(65535), blur_y(di, xi, y, xo));
        Func argmin2("argmin2");
        argmin2(xi, y, xo) = cast<uint16_t>(65535);
        argmin2(xi, y, xo) = min(argmin2(xi, y, xo), second_best(rd, xi, y, xo));
        Func p_clamped("p_clamped");
        p_clamped(xi, y, xo) = clamp(argmin1(xi, y, xo), 1, depth - 2);

        Func subpout(int16, "subpout");
        Expr p = cast<int32_t>(blur_y(cast(int32, p_clamped(xi, y, xo)) + 1, xi, y, xo));
        Expr nn = cast<int32_t>(blur_y(cast(int32, p_clamped(xi, y, xo)) - 1, xi, y, xo));
        Expr d1 = p + nn - 2 * preout(xi, y, xo) + abs(p - nn);
        Expr qv = (abs(p - nn) * 256) / d1;
        Expr quot = select(p >= nn, qv, -qv);
        Expr subpout_expr = cast<int16_t>((cast<int16_t>(depth - p_clamped(xi, y, xo) - 1 + mindisp) * 256 + (select(d1 == 0, 0, quot) + 15)) >> 4);
        subpout(xi, y, xo) = select(argmin1(xi, y, xo) > 0 && argmin1(xi, y, xo) < depth - 1, subpout_expr, cast<int16_t>((depth - argmin1(xi, y, xo) - 1 + mindisp) * 16));

        Expr filtered = cast<int16_t>((mindisp - 1) * 16);
        Expr reject = textblury(xi, y, xo) < threshold;
        if (uniqueness_ratio > 0) {
            reject = reject || (cast<int32_t>(argmin2(xi, y, xo)) <= cast<int32_t>(preout(xi, y, xo)) + (cast<int32_t>(preout(xi, y, xo)) * cast<int32_t>(uniqueness_ratio)) / 100);
        }
        Func splitoutput("splitoutput");
        splitoutput(xi, y, xo) = select(reject, filtered, subpout(xi, y, xo));

        Expr sw2 = winsize / 2;
        Expr in_valid_roi = x >= (mindisp + depth - 1) + sw2 && x < W - mindisp - sw2 && y >= sw2 && y < H - sw2;
        Func output("output");
        output(x, y) = select(in_valid_roi, splitoutput(x % tilesize, y, x / tilesize), filtered);

        proc0.compute_root().vectorize(x, native_lanes * 4);
        proc1.compute_root().vectorize(x, native_lanes * 4);
        xsobel0.compute_root().vectorize(x, native_lanes * 4);
        xsobel1.compute_root().vectorize(x, native_lanes * 4);

        preout.compute_at(splitoutput, xi).update().atomic(false).vectorize(rd, depth);
        argmin1.compute_at(splitoutput, xi).update().atomic(false).vectorize(rd, depth);
        argmin2.compute_at(splitoutput, xi).update().atomic(false).vectorize(rd, depth);

        subpout.compute_at(splitoutput, xi).vectorize(xi, native_lanes);
        splitoutput.compute_root().reorder_storage(xi, xo, y).parallel(xo).vectorize(xi, native_lanes);

        blur_y.bound(di, 0, depth);
        vsum.bound(di, 0, depth);

        // Folded: single-row window. Unfolded ablation: pin to full image height.
        Expr y_window = fold ? Expr(1) : Expr(H);

        blur_y.compute_at(splitoutput, y).store_at(splitoutput, y).vectorize(di, depth);
        vsum.compute_at(splitoutput, y).store_at(splitoutput, xo).vectorize(di, depth).fold_storage(y, y_window);

        f1.compute_at(splitoutput, y).vectorize(di, depth);
        zero_blur.compute_at(splitoutput, xo).vectorize(di, depth);

        zerotext.compute_at(splitoutput, xo).vectorize(xi, native_lanes);
        textsum.compute_at(splitoutput, y).store_at(splitoutput, xo).vectorize(xi).fold_storage(y, y_window);
        textf1.compute_at(splitoutput, y);
        textblury.compute_at(splitoutput, y).store_at(splitoutput, y);

        output.output_buffer().dim(0).set_min(0);
        output.output_buffer().dim(1).set_min(0);
        output.vectorize(x, native_lanes);
        output.compile_jit();
        return output;
    };

    // Schedule 4: the NON-INDUCTIVE window-size-invariant competitor. Both box
    // passes are sliding RDom scans (not inductive funcs), and the disparity axis
    // is split (di innermost) so cost aggregation vectorizes over disparity. It
    // matches the inductive output bit-for-bit but keeps VW*T*N storage per pass
    // (a whole tile of the window) instead of the fold's single row -- the memory
    // cost the paper attributes to doing window-invariance without folding.
    auto build_schedule4 = [&]() -> Func {
        const int T = tilesize, N = ytilesize;
        int VW = vector_width > 0 ? vector_width : native_lanes;
        if (depth % VW != 0) VW = depth;  // require VW | depth
        const int DO = depth / VW;

        Var x("x"), y("y"), di("di"), d_o("d_o");
        Var xi("xi"), xo("xo"), yi("yi"), yo("yo"), d("d");

        Func proc0("s4_proc0"), proc1("s4_proc1");
        proc0(x, y) = cast<int16_t>(BoundaryConditions::mirror_interior(left_gray)(x - winsize / 2, y - winsize / 2));
        proc1(x, y) = cast<int16_t>(BoundaryConditions::mirror_interior(right_gray)(x - winsize / 2, y - winsize / 2));

        Func xsobel0("s4_xsobel0"), xsobel1("s4_xsobel1");
        Expr e0 = proc0(x + 1, y - 1) - proc0(x - 1, y - 1) + 2 * proc0(x + 1, y) - 2 * proc0(x - 1, y) + proc0(x + 1, y + 1) - proc0(x - 1, y + 1);
        Expr e1 = proc1(x + 1, y - 1) - proc1(x - 1, y - 1) + 2 * proc1(x + 1, y) - 2 * proc1(x - 1, y) + proc1(x + 1, y + 1) - proc1(x - 1, y + 1);
        Expr ix = x - winsize / 2, iy = y - winsize / 2;
        Expr border = ix == 0 || ix == W - 1 || ((H % 2 == 1) && iy == H - 1);
        xsobel0(x, y) = select(border, cast<int16_t>(filtercap), cast<int16_t>(clamp(e0, -1 * filtercap, filtercap) + filtercap));
        xsobel1(x, y) = select(border, cast<int16_t>(filtercap), cast<int16_t>(clamp(e1, -1 * filtercap, filtercap) + filtercap));

        Expr ax = xi + xo * T, ay = yi + yo * N;
        Func diff("s4_diff");
        diff(d, xi, yi, xo, yo) = cast<uint16_t>(abs(xsobel0(ax, ay) - xsobel1(ax + d - depth + 1 - mindisp, ay)));

        RDom rwx(0, winsize, "s4_rwx"), rwy(0, winsize, "s4_rwy");
        RDom ryi(1, N - 1, "s4_ryi"), rxi(1, T - 1, "s4_rxi");
        Expr d_full = d_o * VW + di;

        Func vsum(uint16, "s4_vsum");
        vsum(di, xi, yi, xo, yo, d_o) = undef<uint16_t>();
        vsum(di, xi, 0, xo, yo, d_o) = sum(diff(d_full, xi, rwy, xo, yo));
        vsum(di, xi, ryi, xo, yo, d_o) = cast<uint16_t>(vsum(di, xi, ryi - 1, xo, yo, d_o) +
                                                        diff(d_full, xi, ryi + winsize - 1, xo, yo) -
                                                        diff(d_full, xi, ryi - 1, xo, yo));

        Func cSAD(uint16, "s4_cSAD");
        cSAD(di, xi, yi, xo, yo, d_o) = undef<uint16_t>();
        cSAD(di, 0, yi, xo, yo, d_o) = sum(vsum(di, rwx, yi, xo, yo, d_o));
        cSAD(di, rxi, yi, xo, yo, d_o) = cast<uint16_t>(cSAD(di, rxi - 1, yi, xo, yo, d_o) +
                                                        vsum(di, rxi + winsize - 1, yi, xo, yo, d_o) -
                                                        vsum(di, rxi - 1, yi, xo, yo, d_o));

        auto sad = [&](Expr d_expr, Expr X, Expr Y) {
            return cSAD(d_expr % VW, X, Y, xo, yo, d_expr / VW);
        };

        Func text("s4_text"), textcol(uint16, "s4_textcol"), textSAD(uint16, "s4_textSAD");
        text(xi, yi, xo, yo) = cast<uint8_t>(abs(cast<int16_t>(xsobel0(ax, ay)) - cast<int16_t>(filtercap)));
        textcol(xi, yi, xo, yo) = sum(cast<uint16_t>(text(xi + rwx, yi, xo, yo)));
        textSAD(xi, yi, xo, yo) = undef<uint16_t>();
        textSAD(xi, 0, xo, yo) = sum(textcol(xi, rwy, xo, yo));
        textSAD(xi, ryi, xo, yo) = cast<uint16_t>(textSAD(xi, ryi - 1, xo, yo) +
                                                  textcol(xi, ryi + winsize - 1, xo, yo) -
                                                  textcol(xi, ryi - 1, xo, yo));

        RDom rd(0, VW, 0, DO, "s4_rd");
        Func preout("s4_preout");
        preout(xi, yi, xo, yo) = cast<uint16_t>(65535);
        preout(xi, yi, xo, yo) = min(cSAD(rd[0], xi, yi, xo, yo, rd[1]), preout(xi, yi, xo, yo));
        Func prearg("s4_prearg");
        prearg(di, xi, yi, xo, yo, d_o) = select(preout(xi, yi, xo, yo) == cSAD(di, xi, yi, xo, yo, d_o), cast<uint16_t>(d_full), cast<uint16_t>(65535));
        Func argmin1("s4_argmin1");
        argmin1(xi, yi, xo, yo) = cast<uint16_t>(65535);
        argmin1(xi, yi, xo, yo) = min(argmin1(xi, yi, xo, yo), prearg(rd[0], xi, yi, xo, yo, rd[1]));
        Func second_best("s4_second_best");
        second_best(di, xi, yi, xo, yo, d_o) = select(abs(cast<int16_t>(d_full) - cast<int16_t>(argmin1(xi, yi, xo, yo))) <= 1, cast<uint16_t>(65535), cSAD(di, xi, yi, xo, yo, d_o));
        Func argmin2("s4_argmin2");
        argmin2(xi, yi, xo, yo) = cast<uint16_t>(65535);
        argmin2(xi, yi, xo, yo) = min(argmin2(xi, yi, xo, yo), second_best(rd[0], xi, yi, xo, yo, rd[1]));
        Func p_clamped("s4_p_clamped");
        p_clamped(xi, yi, xo, yo) = clamp(argmin1(xi, yi, xo, yo), 1, depth - 2);

        Func subpout(int16, "s4_subpout");
        Expr pc = cast<int32_t>(p_clamped(xi, yi, xo, yo));
        Expr p = cast<int32_t>(sad(pc + 1, xi, yi));
        Expr nn = cast<int32_t>(sad(pc - 1, xi, yi));
        Expr d1 = p + nn - 2 * preout(xi, yi, xo, yo) + abs(p - nn);
        Expr qv = (abs(p - nn) * 256) / d1;
        Expr quot = select(p >= nn, qv, -qv);
        Expr subpout_expr = cast<int16_t>((cast<int16_t>(depth - p_clamped(xi, yi, xo, yo) - 1 + mindisp) * 256 + (select(d1 == 0, 0, quot) + 15)) >> 4);
        subpout(xi, yi, xo, yo) = select(argmin1(xi, yi, xo, yo) > 0 && argmin1(xi, yi, xo, yo) < depth - 1, subpout_expr, cast<int16_t>((depth - argmin1(xi, yi, xo, yo) - 1 + mindisp) * 16));

        Expr filtered = cast<int16_t>((mindisp - 1) * 16);
        Expr reject = textSAD(xi, yi, xo, yo) < threshold;
        if (uniqueness_ratio > 0) {
            reject = reject || (cast<int32_t>(argmin2(xi, yi, xo, yo)) <= cast<int32_t>(preout(xi, yi, xo, yo)) + (cast<int32_t>(preout(xi, yi, xo, yo)) * cast<int32_t>(uniqueness_ratio)) / 100);
        }
        Func disp_left("s4_disp_left");
        disp_left(xi, yi, xo, yo) = select(reject, filtered, subpout(xi, yi, xo, yo));

        Expr sw2 = winsize / 2;
        Expr in_valid_roi = x >= (mindisp + depth - 1) + sw2 && x < W - mindisp - sw2 && y >= sw2 && y < H - sw2;
        Func output("s4_output");
        output(x, y) = select(in_valid_roi, disp_left(x % T, y % N, x / T, y / N), filtered);

        proc0.compute_root().vectorize(x, native_lanes * 4);
        proc1.compute_root().vectorize(x, native_lanes * 4);
        xsobel0.compute_root().vectorize(x, native_lanes * 4);
        xsobel1.compute_root().vectorize(x, native_lanes * 4);

        disp_left.compute_root().reorder_storage(xi, yi, xo, yo);
        disp_left.reorder(xi, yi, xo, yo).parallel(yo).parallel(xo).vectorize(xi, native_lanes);

        preout.compute_at(disp_left, xi).update().atomic(false).reorder(rd[0], xi, yi, rd[1]).vectorize(rd[0], VW);
        argmin1.compute_at(disp_left, xi).update().atomic(false).reorder(rd[0], xi, yi, rd[1]).vectorize(rd[0], VW);
        argmin2.compute_at(disp_left, xi).update().atomic(false).reorder(rd[0], xi, yi, rd[1]).vectorize(rd[0], VW);
        subpout.compute_at(disp_left, xi).vectorize(xi, native_lanes);

        cSAD.compute_at(disp_left, xo).reorder_storage(di, xi, yi, xo, yo, d_o);
        cSAD.reorder(di, xi, yi, d_o).vectorize(di, VW);
        cSAD.update(0).reorder(di, yi, d_o).vectorize(di, VW);
        cSAD.update(1).reorder(di, rxi, yi, d_o).vectorize(di, VW);

        vsum.compute_at(disp_left, xo).reorder_storage(di, xi, yi, xo, yo, d_o);
        vsum.reorder(di, xi, yi, d_o).vectorize(di, VW);
        vsum.update(0).reorder(di, xi, d_o).vectorize(di, VW);
        vsum.update(1).reorder(di, xi, ryi, d_o).vectorize(di, VW);

        textSAD.compute_at(disp_left, xo).vectorize(xi, native_lanes);
        textSAD.update(0).vectorize(xi, native_lanes);
        textSAD.update(1).vectorize(xi, native_lanes);
        textcol.compute_at(disp_left, xo).vectorize(xi, native_lanes);

        output.output_buffer().dim(0).set_min(0);
        output.output_buffer().dim(1).set_min(0);
        output.vectorize(x, native_lanes);
        output.compile_jit();
        return output;
    };

    Func fold_pipe, unfold_pipe, s4_pipe;
    try {
        fold_pipe = build(true);
        unfold_pipe = build(false);
        s4_pipe = build_schedule4();
    } catch (const Halide::Error &e) {
        printf("BUILD ERROR: %s\n", e.what());
        return 3;
    }

    Buffer<int16_t> of(W, H), ou(W, H), os4(W, H);
    fold_pipe.realize(of);
    unfold_pipe.realize(ou);
    s4_pipe.realize(os4);

    hb::Stats s_fold = hb::bench([&] { fold_pipe.realize(of); });
    hb::Stats s_unfold = hb::bench([&] { unfold_pipe.realize(ou); });
    hb::Stats s_s4 = hb::bench([&] { s4_pipe.realize(os4); });

    // Measured peak internal-heap footprint per variant (untimed, custom
    // allocator; output images are user-supplied, allocated outside realize, so
    // only the pipeline's SAD-window scratch is counted).
    const double meas_fold = hb::measure_jit_peak(fold_pipe, [&] { fold_pipe.realize(of); });
    const double meas_unf = hb::measure_jit_peak(unfold_pipe, [&] { unfold_pipe.realize(ou); });
    const double meas_s4 = hb::measure_jit_peak(s4_pipe, [&] { s4_pipe.realize(os4); });

    // Correctness: folded and unfolded must be bit-identical (same math). Schedule4
    // is the non-inductive variant and must match too (bit-for-bit per its design).
    size_t mism = 0, mism_s4 = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            if (of(x, y) != ou(x, y)) mism++;
            if (of(x, y) != os4(x, y)) mism_s4++;
        }
    bool ok = (mism == 0);

    // Roofline x-axis: unfolded footprint of the column-inductive buffers
    // (vsum: depth*tilesize*H, textsum: tilesize*H, uint16) that folding removes.
    const double bytes_unf = ((double)depth * tilesize * H + (double)tilesize * H) * 2.0;
    const double bytes_fold = ((double)depth * tilesize + (double)tilesize) * 2.0;
    const double mpix = (double)W * H / 1e6;
    char note[200];
    snprintf(note, sizeof(note),
             "StereoBM (JIT)  W=%d H=%d winsize=%d depth=%d tilesize=%d  |  state fold %.0fx  |  unfolded fp/LLC=%.3f",
             W, H, winsize, depth, tilesize, hb::mem_ratio(bytes_unf, bytes_fold),
             hb::footprint_over_llc(bytes_unf));
    hb::print_spec_header("stereobm_jit", "host", note);

#if STEREOBM_BUILD_OPENCV
    // OpenCV StereoBM on the SAME image + params, same core count -> timing baseline.
    cv::setNumThreads(num_threads);
    cv::Mat cvL(H, W, CV_8U), cvR(H, W, CV_8U);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            cvL.at<uchar>(y, x) = lg(x, y);
            cvR.at<uchar>(y, x) = rg(x, y);
        }
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(depth, winsize);
    bm->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
    bm->setPreFilterCap(filtercap);
    bm->setMinDisparity(mindisp);
    bm->setTextureThreshold(threshold);
    bm->setUniquenessRatio(uniqueness_ratio);
    cv::Mat cv_disp;
    hb::Stats s_cv = hb::bench([&] { bm->compute(cvL, cvR, cv_disp); });
    hb::print_row("OpenCV StereoBM (baseline)", s_cv, mpix / (s_cv.min * 1e-3),
                  "Mpix/s", 0.0, -1.0, true);
#endif

    hb::print_row("inductive UNFOLDED (fold y -> H)", s_unfold, mpix / (s_unfold.min * 1e-3),
                  "Mpix/s", meas_unf, (double)mism, ok,
#if STEREOBM_BUILD_OPENCV
                  hb::verdict(s_unfold.min, s_cv.min),
#else
                  "",
#endif
                  bytes_unf);
    hb::print_row("inductive FOLDED (fold y -> 1)", s_fold, mpix / (s_fold.min * 1e-3),
                  "Mpix/s", meas_fold, (double)mism, ok, hb::verdict(s_fold.min, s_unfold.min), bytes_unf);
    // Schedule4: non-inductive sliding-RDom competitor. state = VW*T*N per pass
    // (vsum + cSAD), the tile-sized window storage folding avoids. Verdict is vs
    // the inductive FOLDED variant (the whole point: same result, more memory).
    hb::print_row("schedule4 (non-inductive, RDom slide)", s_s4, mpix / (s_s4.min * 1e-3),
                  "Mpix/s", meas_s4, (double)mism_s4, mism_s4 == 0,
                  hb::verdict(s_fold.min, s_s4.min), bytes_unf);
    return (ok && mism_s4 == 0) ? 0 : 1;
}
