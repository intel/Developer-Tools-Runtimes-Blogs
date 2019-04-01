/* ============================================================================
 * Copyright (C) 2019 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 * OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 * SPDX-License-Identifier: MIT
 * ============================================================================
 */

#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <algorithm>
#include <immintrin.h>

//Time
#include <Windows.h>
#define rdtsc __rdtsc

#define testIterations 100000

static int preventOptimize = 0;

static const int length = 1024*8;
static float a[length];

static float scalarAverage() {
    preventOptimize++;
    float sum = 0.0;
    for (int i = 0; i < _countof(a); i++) {
        sum += a[i];
    }
    return sum / _countof(a);
}

__declspec(noinline) static float avxAverage() {
    preventOptimize++;
    __m256 sumx8 = _mm256_setzero_ps();
    for (int i = 0; i < _countof(a); i = i + 8) {
        sumx8 = _mm256_add_ps(sumx8, _mm256_loadu_ps(&(a[i])));
    }
    float sum = sumx8.m256_f32[0] + sumx8.m256_f32[1] +
        sumx8.m256_f32[2] + sumx8.m256_f32[3] +
        sumx8.m256_f32[4] + sumx8.m256_f32[5] +
        sumx8.m256_f32[6] + sumx8.m256_f32[7];
    return sum / _countof(a);
}

__declspec(noinline) static float avx512Average() {
    preventOptimize++;
    __m512 sumx16 = _mm512_setzero_ps();
    for (int i = 0; i < _countof(a); i = i + 16) {
        sumx16 = _mm512_add_ps(sumx16, _mm512_loadu_ps(&(a[i])));
    }
    float sum = _mm512_reduce_add_ps(sumx16);
    return sum / _countof(a);
}

static bool sanityCheck() {
    float scalarVal = scalarAverage();
    float avxVal = avxAverage();
    float avx512Val = avx512Average();

    return fabs(avxVal - scalarVal) < 0.0001 &&
        fabs(avx512Val - scalarVal) < 0.0001;
}

static bool initArray() {
    for (int i = 0; i < length; ++i) {
        a[i] = 0.1f;
    }
    // Check that the two kernel functions yields the same result.
    return sanityCheck();
}

int main(void) {
    static uint64_t start, end;
    double dtAvx, dtAvx512;
    float val = 0.0f;
    if (!initArray()) {
        printf("Wrong implementation.\n");
        return 0;
    }
    // AVX2
    start = rdtsc();
    for (int i = 0; i < testIterations; ++i) {
        val += avxAverage();
    }
    end = rdtsc();
    dtAvx = end - start;

    // AVX512
    start = rdtsc();
    for (int i = 0; i < testIterations; ++i) {
        val -= avx512Average();
    }
    end = rdtsc();
    dtAvx512 = end - start;
    if (fabs(val) > 0.01) {
        printf("Wrong result. The difference is %f\n", val);
        return 0;
    }
    printf("AVX2/AVX512 = %f", dtAvx / dtAvx512);
    return preventOptimize;
}
