// Third-party GPU baseline for the batched prefix scan: NVIDIA Thrust/CUB via
// thrust::inclusive_scan_by_key (a segmented inclusive scan -- one independent
// prefix sum per lane). Reads the {W,S,input,output} file dumped by
// prefixsum_gpu.cpp, times the vendor scan with the same best-of protocol, and
// validates its result against Halide's output (bit-for-bit; integer sums).
//
// Build:  nvcc -O3 -std=c++17 prefixsum_gpu_thrust.cu -o prefixsum_gpu_thrust
// Run:    ./prefixsum_gpu_thrust /tmp/prefixsum_gpu_data.bin

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

static int env_int(const char *k, int d) {
    const char *v = getenv(k);
    return (v && *v) ? atoi(v) : d;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/prefixsum_gpu_data.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    int32_t hdr[2];
    if (fread(hdr, sizeof(int32_t), 2, f) != 2) { fprintf(stderr, "bad header\n"); return 1; }
    const int W = hdr[0], S = hdr[1];
    const size_t N = (size_t)W * S;
    std::vector<int32_t> in(N), ref(N);
    if (fread(in.data(), sizeof(int32_t), N, f) != N ||
        fread(ref.data(), sizeof(int32_t), N, f) != N) {
        fprintf(stderr, "bad body\n"); return 1;
    }
    fclose(f);

    // Segment keys: one key value per lane (contiguous, lane-major layout).
    std::vector<int32_t> keys(N);
    for (int s = 0; s < S; s++)
        for (int t = 0; t < W; t++) keys[(size_t)s * W + t] = s;

    thrust::device_vector<int32_t> d_in(in), d_keys(keys), d_out(N);

    auto scan = [&] {
        thrust::inclusive_scan_by_key(thrust::device, d_keys.begin(), d_keys.end(),
                                      d_in.begin(), d_out.begin());
    };

    const int warmup = env_int("HB_WARMUP", 3);
    const int trials = std::max(1, env_int("HB_TRIALS", 30));
    for (int i = 0; i < warmup; i++) scan();
    cudaDeviceSynchronize();

    std::vector<double> ms;
    ms.reserve(trials);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < trials; i++) {
        cudaEventRecord(a);
        scan();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float e = 0; cudaEventElapsedTime(&e, a, b);
        ms.push_back(e);
    }
    std::sort(ms.begin(), ms.end());
    const double best = ms.front(), median = ms[trials / 2];

    std::vector<int32_t> out(N);
    thrust::copy(d_out.begin(), d_out.end(), out.begin());
    size_t mism = 0;
    for (size_t i = 0; i < N; i++) if (out[i] != ref[i]) mism++;

    const double mpix = (W * (double)S) / (best * 1e3);

    // --- CUB DeviceScan::InclusiveSum: the single-array decoupled-look-back scan
    // (CUB's flagship, one pass). Different SEMANTICS -- it scans the whole N-element
    // array as ONE sequence, crossing lane boundaries, so its result is NOT the
    // segmented output and is not validated against ref; it is the peak-scan-
    // throughput reference only. Same total element count and same read+write
    // traffic, so its Mpix/s is directly comparable as an upper bound. ---
    thrust::device_vector<int32_t> d_out2(N);
    void *d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes,
        thrust::raw_pointer_cast(d_in.data()),
        thrust::raw_pointer_cast(d_out2.data()), (int)N);
    cudaMalloc(&d_temp, temp_bytes);
    auto cub_scan = [&] {
        cub::DeviceScan::InclusiveSum(d_temp, temp_bytes,
            thrust::raw_pointer_cast(d_in.data()),
            thrust::raw_pointer_cast(d_out2.data()), (int)N);
    };
    for (int i = 0; i < warmup; i++) cub_scan();
    cudaDeviceSynchronize();
    std::vector<double> ms2; ms2.reserve(trials);
    for (int i = 0; i < trials; i++) {
        cudaEventRecord(a); cub_scan(); cudaEventRecord(b);
        cudaEventSynchronize(b);
        float e = 0; cudaEventElapsedTime(&e, a, b); ms2.push_back(e);
    }
    std::sort(ms2.begin(), ms2.end());
    const double cub_best = ms2.front(), cub_median = ms2[trials / 2];
    const double cub_mpix = (W * (double)S) / (cub_best * 1e3);
    cudaFree(d_temp);

    printf("### prefixsum_gpu_thrust (third-party GPU baselines: NVIDIA CUB / Thrust)\n");
    printf("### protocol: warmup=%d trials=%d  metric=best(min) ms  |  W=%d S=%d  N=%zu\n",
           warmup, trials, W, S, N);
    printf("  %-38s %10.3f %10.3f   %8.1f %-4s  %s (err %zu)\n",
           "Thrust inclusive_scan_by_key (segmented)", best, median, mpix, "Mpix/s",
           mism == 0 ? "PASS" : "FAIL", mism);
    printf("  %-38s %10.3f %10.3f   %8.1f %-4s  %s\n",
           "CUB DeviceScan::InclusiveSum (1 array)", cub_best, cub_median, cub_mpix, "Mpix/s",
           "peak-ref (unsegmented)");
    printf("  %s\n", mism == 0 ? "Success!" : "FAILED (differs from Halide)");
    return mism == 0 ? 0 : 1;
}
