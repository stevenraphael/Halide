// M*//////////////////////////////////////////////////////////////////////////////////////
//
//   IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//   By downloading, copying, installing or using the software you agree to this license.
//   If you do not agree to this license, do not download, install,
//   copy or use the software.
//
//
//                           License Agreement
//                 For Open Source Computer Vision Library
//
//  Copyright (C) 2000, Intel Corporation, all rights reserved.
//  Copyright (C) 2013, OpenCV Foundation, all rights reserved.
//  Third party copyrights are property of their respective owners.
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
//
//    * Redistribution's of source code must retain the above copyright notice,
//      this list of conditions and the following disclaimer.
//
//    * Redistribution's in binary form must reproduce the above copyright notice,
//      this list of conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//
//    * The name of the copyright holders may not be used to endorse or promote products
//      derived from this software without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors "as is" and
//  any express or implied warranties, including, but not limited to, the implied
//  warranties of merchantability and fitness for a particular purpose are disclaimed.
//  In no event shall the Intel Corporation or contributors be liable for any direct,
//  indirect, incidental, special, exemplary, or consequential damages
//  (including, but not limited to, procurement of substitute goods or services;
//  loss of use, data, or profits; or business interruption) however caused
//  and on any theory of liability, whether in contract, strict liability,
//  or tort (including negligence or otherwise) arising in any way out of
//  the use of this software, even if advised of the possibility of such damage.
//
// M*/

// Halide adaptation of https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/stereobm.cpp
//
// This is a variant of stereobm_generator.cpp that does NOT use inductive
// (self-referential) functions for the box filter. Instead, following the
// "Schedule 3" formulation, the window sums are expressed as plain sum()
// reductions over RDoms. This achieves the same locality/parallelism but
// redundantly recomputes the window sum at every pixel, so runtime grows with
// the window size. The output is bit-for-bit identical to stereobm_generator.

#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

class StereoBMTiled : public Generator<StereoBMTiled> {
public:
    // Inputs: two grayscale images of the same scene as seen by a left and right camera
    Input<Buffer<uint8_t, 2>> left_gray{"left_gray"};
    Input<Buffer<uint8_t, 2>> right_gray{"right_gray"};

    GeneratorParam<int> winsize{"winsize", 9};                    // size of block surrounding each pixel to compare; must be odd
    GeneratorParam<int> depth{"depth", 16};                       // maximum number of disparities to consider
    GeneratorParam<int> tilesize{"tilesize", 64};                 // strip size (x tile)
    GeneratorParam<int> ytilesize{"ytilesize", 64};               // y tile size for the cSAD scan
    GeneratorParam<int> threshold{"threshold", 10};               // reject pixels if sum of prefiltered pixels in block is less than this
    GeneratorParam<int> mindisp{"mindisp", 0};                    // minimum disparity to consider.
    GeneratorParam<int> uniqueness_ratio{"uniqueness_ratio", 0};  // reject if the second-best match is too close to the best match, as a percentage of the best match score.
    GeneratorParam<int> filtercap{"filtercap", 31};               // clamp the prefiltering output to the range [0, filtercap*2].
    Output<Buffer<int16_t, 2>> output{"output"};

    void generate() {
        const Type uint16 = UInt(16);
        const Type int16 = Int(16);
        const Type int32 = Int(32);
        const int native_lanes = get_target().natural_vector_size<int16_t>();

        Var y("y");
        Var x("x");
        Var c("c");
        Var di("di");
        Var xi("xi"), xo("xo");

        Expr W = left_gray.dim(0).extent();
        Expr H = left_gray.dim(1).extent();

        Func proc0("proc0"), proc1("proc1");
        // shift the input to ensure the indices of blur_y correspond to the center pixel in the window.
        proc0(x, y) = cast<int16_t>(BoundaryConditions::mirror_interior(left_gray)(x - winsize / 2, y - winsize / 2));
        proc1(x, y) = cast<int16_t>(BoundaryConditions::mirror_interior(right_gray)(x - winsize / 2, y - winsize / 2));

        // prefiltering with sobel filter
        Func xsobel0("xsobel0"), xsobel1("xsobel1");
        Expr e0 = proc0(x + 1, y - 1) - proc0(x - 1, y - 1) + 2 * proc0(x + 1, y) - 2 * proc0(x - 1, y) + proc0(x + 1, y + 1) - proc0(x - 1, y + 1);
        Expr e1 = proc1(x + 1, y - 1) - proc1(x - 1, y - 1) + 2 * proc1(x + 1, y) - 2 * proc1(x - 1, y) + proc1(x + 1, y + 1) - proc1(x - 1, y + 1);
        Expr ix = x - winsize / 2, iy = y - winsize / 2;  // image-space coordinates
        Expr border = ix == 0 || ix == W - 1 || ((H % 2 == 1) && iy == H - 1);
        xsobel0(x, y) = select(border, cast<int16_t>(filtercap), cast<int16_t>(clamp(e0, -1 * filtercap, filtercap) + filtercap));
        xsobel1(x, y) = select(border, cast<int16_t>(filtercap), cast<int16_t>(clamp(e1, -1 * filtercap, filtercap) + filtercap));

        // The y axis is tiled the same way the x axis already is: an outer tile index yo
        // and an inner index yi, with absolute row = yo * N + yi. Every function below
        // carries (xi, yi, xo, yo) instead of a flat y, so no y%N / y/N appears in any
        // producer index -- the split is threaded through explicitly, exactly like xi/xo.
        const int N = ytilesize;
        Var yi("yi"), yo("yo");
        Expr ax = xi + xo * tilesize;  // absolute x
        Expr ay = yi + yo * N;         // absolute y

        Func diff("diff");
        // for each disparity di, sample right image (depth-1-di)+mindisp pixels left of the left image's pixel
        // image is divided into strips indexed by (xo, yo) that are processed in parallel
        diff(di, xi, yi, xo, yo) = Halide::cast<uint16_t>(cast<uint16_t>(abs((xsobel0(ax, ay)) - (xsobel1(ax + di - depth + 1 - mindisp, ay)))));

        // --- SAD box filter, expressed with RDom scans over y tiles (no inductive funcs) ---
        // vsum is the horizontal (x) window sum of diff (a plain sum, hence the redundant
        // recomputation of Schedule 3). cSAD is the vertical window sum of vsum, computed
        // as a scan within each y tile: the base row (yi==0) is a full window sum, and each
        // subsequent row slides the window with an RDom update over the tile's rows.
        RDom rwx(0, winsize, "rwx");  // horizontal window
        RDom rwy(0, winsize, "rwy");  // vertical window (tile base row)
        RDom ryi(1, N - 1, "ryi");    // rows within a tile, after the base row

        Func vsum(uint16, "vsum");
        vsum(di, xi, yi, xo, yo) = sum(cast<uint16_t>(diff(di, xi + rwx, yi, xo, yo)));

        Func cSAD(uint16, "cSAD");
        cSAD(di, xi, yi, xo, yo) = undef<uint16_t>();
        // base row of the tile: full vertical window sum over tile rows [0, winsize)
        cSAD(di, xi, 0, xo, yo) = sum(vsum(di, xi, rwy, xo, yo));
        // slide the window down through the rest of the tile
        cSAD(di, xi, ryi, xo, yo) = cast<uint16_t>(cSAD(di, xi, ryi - 1, xo, yo) +
                                                   vsum(di, xi, ryi + winsize - 1, xo, yo) -
                                                   vsum(di, xi, ryi - 1, xo, yo));

        // compute the texture value for each pixel, which is the sum of pixels in the surrounding block
        Func text("text");
        Func textcol(uint16, "textcol");
        Func textSAD(uint16, "textSAD");
        text(xi, yi, xo, yo) = cast<uint8_t>(abs(cast<int16_t>(xsobel0(ax, ay)) - cast<int16_t>(filtercap)));
        textcol(xi, yi, xo, yo) = sum(cast<uint16_t>(text(xi + rwx, yi, xo, yo)));
        textSAD(xi, yi, xo, yo) = undef<uint16_t>();
        textSAD(xi, 0, xo, yo) = sum(textcol(xi, rwy, xo, yo));
        textSAD(xi, ryi, xo, yo) = cast<uint16_t>(textSAD(xi, ryi - 1, xo, yo) +
                                                  textcol(xi, ryi + winsize - 1, xo, yo) -
                                                  textcol(xi, ryi - 1, xo, yo));

        // compute the best and second-best disparity for each pixel
        Func preout("preout");
        RDom rd(0, depth, "rd");
        preout(xi, yi, xo, yo) = cast<uint16_t>(65535);
        preout(xi, yi, xo, yo) = min(cSAD(rd, xi, yi, xo, yo), preout(xi, yi, xo, yo));
        Func prearg("prearg");
        prearg(di, xi, yi, xo, yo) = select(preout(xi, yi, xo, yo) == cSAD(di, xi, yi, xo, yo), cast<uint16_t>(di), cast<uint16_t>(65535));
        Func argmin1("argmin1");
        argmin1(xi, yi, xo, yo) = cast<uint16_t>(65535);
        argmin1(xi, yi, xo, yo) = min(argmin1(xi, yi, xo, yo), prearg(rd, xi, yi, xo, yo));
        Func second_best("second_best");
        second_best(di, xi, yi, xo, yo) = select(abs(di - cast<int16_t>(argmin1(xi, yi, xo, yo))) <= 1, cast<uint16_t>(65535), cSAD(di, xi, yi, xo, yo));
        Func argmin2("argmin2");
        argmin2(xi, yi, xo, yo) = cast<uint16_t>(65535);
        argmin2(xi, yi, xo, yo) = min(argmin2(xi, yi, xo, yo), second_best(rd, xi, yi, xo, yo));
        Func p_clamped("p_clamped");
        p_clamped(xi, yi, xo, yo) = clamp(argmin1(xi, yi, xo, yo), 1, depth - 2);  // indexes into cSAD

        // subpixel refinement
        Func subpout(int16, "subpout");
        Expr p = cast<int32_t>(cSAD(cast(int32, p_clamped(xi, yi, xo, yo)) + 1, xi, yi, xo, yo));
        Expr n = cast<int32_t>(cSAD(cast(int32, p_clamped(xi, yi, xo, yo)) - 1, xi, yi, xo, yo));
        Expr d1 = p + n - 2 * preout(xi, yi, xo, yo) + abs(p - n);
        Expr q = (abs(p - n) * 256) / d1;
        Expr quot = select(p >= n, q, -q);
        Expr subpout_expr = cast<int16_t>((cast<int16_t>(depth - p_clamped(xi, yi, xo, yo) - 1 + mindisp) * 256 + (select(d1 == 0, 0, quot) + 15)) >> 4);
        subpout(xi, yi, xo, yo) = select(argmin1(xi, yi, xo, yo) > 0 && argmin1(xi, yi, xo, yo) < depth - 1, subpout_expr, cast<int16_t>((depth - argmin1(xi, yi, xo, yo) - 1 + mindisp) * 16));

        // edge case handling
        Expr filtered = cast<int16_t>((mindisp - 1) * 16);
        Expr reject = textSAD(xi, yi, xo, yo) < threshold;  // reject if block texture is too uniform
        if (int(uniqueness_ratio) > 0) {                    // reject if second-best disparity is too close to best disparity
            reject = reject || (cast<int32_t>(argmin2(xi, yi, xo, yo)) <= cast<int32_t>(preout(xi, yi, xo, yo)) + (cast<int32_t>(preout(xi, yi, xo, yo)) * cast<int32_t>(uniqueness_ratio)) / 100);
        }
        Func splitoutput("splitoutput");
        splitoutput(xi, yi, xo, yo) = select(reject, filtered, subpout(xi, yi, xo, yo));

        // disparities are only valid where the SAD window and the full disparity search both fit inside the image.
        Expr sw2 = winsize / 2;
        Expr in_valid_roi = x >= (static_cast<int>(mindisp) + static_cast<int>(depth) - 1) + sw2 && x < W - static_cast<int>(mindisp) - sw2 &&
                            y >= sw2 && y < H - sw2;
        output(x, y) = select(in_valid_roi, splitoutput(x % tilesize, y % N, x / tilesize, y / N), filtered);

        // --- Schedule ---
        proc0.compute_root().vectorize(x, native_lanes * 4);
        proc1.compute_root().vectorize(x, native_lanes * 4);
        xsobel0.compute_root().vectorize(x, native_lanes * 4);
        xsobel1.compute_root().vectorize(x, native_lanes * 4);

        // splitoutput is indexed by (xi, yi, xo, yo). The tile loops (xo, yo) are outermost
        // and parallel; the per-tile scans are computed once per (xo, yo) tile.
        splitoutput.compute_root().reorder_storage(xi, xo, yi, yo);
        splitoutput.reorder(xi, yi, xo, yo)
            .parallel(yo)
            .parallel(xo)
            .vectorize(xi, native_lanes);

        preout.compute_at(splitoutput, xi).update().atomic(false).vectorize(rd, depth);
        argmin1.compute_at(splitoutput, xi).update().atomic(false).vectorize(rd, depth);
        argmin2.compute_at(splitoutput, xi).update().atomic(false).vectorize(rd, depth);
        subpout.compute_at(splitoutput, xi).vectorize(xi, native_lanes);

        // Compute each tile's box-filter scan once, reused across the tile's rows.
        cSAD.compute_at(splitoutput, xo).vectorize(di, depth);
        cSAD.update(0).vectorize(di, depth);
        cSAD.update(1).vectorize(di, depth);
        vsum.compute_at(splitoutput, xo).reorder(di, xi).vectorize(di, depth);

        textSAD.compute_at(splitoutput, xo).vectorize(xi, native_lanes);
        textSAD.update(0).vectorize(xi, native_lanes);
        textSAD.update(1).vectorize(xi, native_lanes);
        textcol.compute_at(splitoutput, xo).vectorize(xi, native_lanes);

        output.dim(0).set_min(0);
        output.dim(1).set_min(0);

        output.vectorize(x, native_lanes);
    }
};

HALIDE_REGISTER_GENERATOR(StereoBMTiled, stereobm)
