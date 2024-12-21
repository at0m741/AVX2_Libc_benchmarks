#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

void *_memcpy_avx(void *dest, const void *src, size_t len) {
    if (len == 0 || dest == src) {
        return dest;
    }

    uint8_t *cdest = (uint8_t *)dest;
    const uint8_t *csrc = (const uint8_t *)src;

    size_t num_blocks = len / 32; 
    size_t remaining_bytes = len % 32;

    size_t num_unrolled = num_blocks / 8;  
    size_t remainder_blocks = num_blocks % 8;

    const __m256i *s = (const __m256i *)csrc;
    __m256i *d = (__m256i *)cdest;

    for (size_t i = 0; i < num_unrolled; ++i) {
	if (len <= 256){
		_mm_prefetch((const char *)(s + 8), _MM_HINT_T0);
		_mm_prefetch((const char *)(d + 8), _MM_HINT_T0);
	}
	_mm256_storeu_si256(d++, _mm256_loadu_si256(s++));
        _mm256_storeu_si256(d++, _mm256_loadu_si256(s++));
        _mm256_storeu_si256(d++, _mm256_loadu_si256(s++));
        _mm256_storeu_si256(d++, _mm256_loadu_si256(s++));
        _mm256_storeu_si256(d++, _mm256_loadu_si256(s++));
        _mm256_storeu_si256(d++, _mm256_loadu_si256(s++));
	_mm256_storeu_si256(d++, _mm256_loadu_si256(s++));
        _mm256_storeu_si256(d++, _mm256_loadu_si256(s++));
    }	
    for (size_t i = 0; i < remainder_blocks; ++i) {
        _mm256_storeu_si256(d++, _mm256_loadu_si256(s++));
    }

    uint8_t *byte_dest = (uint8_t *)d;
    const uint8_t *byte_src = (const uint8_t *)s;
    for (size_t i = 0; i < remaining_bytes; ++i) {
        byte_dest[i] = byte_src[i];
    }

    return dest;
}


#include <stdlib.h>
#include <string.h>
#include <time.h>
double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

void benchmark_memcpy(size_t size, size_t iterations) {
    uint8_t *src = (uint8_t *)aligned_alloc(32, size);
    uint8_t *dest_std = (uint8_t *)aligned_alloc(32, size);
    uint8_t *dest_custom = (uint8_t *)aligned_alloc(32, size);

    if (!src || !dest_std || !dest_custom) {
        fprintf(stderr, "Error alloc\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < size; ++i) {
        src[i] = rand() % 256;
    }

    double start, end;
    double time_std, time_custom;

    start = get_time_sec();
    for (size_t i = 0; i < iterations; ++i) {
        memcpy(dest_std, src, size);
    }
    end = get_time_sec();
    time_std = end - start;

    start = get_time_sec();
    for (size_t i = 0; i < iterations; ++i) {
        _memcpy_avx(dest_custom, src, size);
    }
    end = get_time_sec();
    time_custom = end - start;

    if (memcmp(dest_std, dest_custom, size) != 0) {
        fprintf(stderr, "Error: different results of memcpy and _memcpy_avx for sizeof %zu\n", size);
    } else {
        printf("Size: %8zu bytes | memcpy: %8.6f s | _memcpy_avx: %8.6f s | Gain: %.2f%%\n",
               size, time_std, time_custom,
               ((time_std - time_custom) / time_std) * 100);
    }

    free(src);
    free(dest_std);
    free(dest_custom);
}

int main() {
    srand(time(NULL));

    size_t sizes[] = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    size_t iterations = 100000;

    printf("Benchmark of memcpy vs _memcpy_avx\n");
    printf("Iter: %zu\n\n", iterations);
    printf("Size (bytes) | memcpy (s) | _memcpy_avx (s) | Gain\n");
    printf("---------------------------------------------------------\n");

    for (size_t i = 0; i < num_sizes; ++i) {
        benchmark_memcpy(sizes[i], iterations);
    }

    return 0;
}
