#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void *_memmove(void *dest, const void *src, size_t len) {
    if (len == 0 || dest == src) {
        return dest;
    }

    uint8_t *cdest = (uint8_t *)dest;
    const uint8_t *csrc = (const uint8_t *)src;

    if (cdest < csrc) {
        size_t num_blocks = len / 32;
        size_t remaining_bytes = len % 32;

        const __m256i *s = (const __m256i *)csrc;
        __m256i *d = (__m256i *)cdest;
		
        for (size_t i = 0; i < num_blocks; ++i) {
            _mm256_storeu_si256(d++, _mm256_loadu_si256(s++));
        }

        uint8_t *byte_dest = (uint8_t *)d;
        const uint8_t *byte_src = (const uint8_t *)s;
        for (size_t i = 0; i < remaining_bytes; ++i) {
            byte_dest[i] = byte_src[i];
        }
    } else {
        size_t num_blocks = len / 32;
        size_t remaining_bytes = len % 32;

        const __m256i *s = (const __m256i *)(csrc + len);
        __m256i *d = (__m256i *)(cdest + len);

        if (remaining_bytes) {
            const uint8_t *byte_src = csrc + len;
            uint8_t *byte_dest = cdest + len;
            for (size_t i = 0; i < remaining_bytes; ++i) {
                byte_dest[-(int)(i + 1)] = byte_src[-(int)(i + 1)];
            }
            s = (const __m256i *)(csrc + len - remaining_bytes);
            d = (__m256i *)(cdest + len - remaining_bytes);
        }

        for (size_t i = 0; i < num_blocks; ++i) {
            d--;
            s--;
            _mm256_storeu_si256(d, _mm256_loadu_si256(s));
        }
    }

    return dest;
}

double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}
void benchmark_memmove(size_t size, size_t iterations) {
    uint8_t *src = (uint8_t *)aligned_alloc(32, size);
    uint8_t *dest_std = (uint8_t *)aligned_alloc(32, size);
    uint8_t *dest_custom = (uint8_t *)aligned_alloc(32, size);

    if (!src || !dest_std || !dest_custom) {
        fprintf(stderr, "Erreur d'allocation mémoire\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < size; ++i) {
        src[i] = rand() % 256;
    }

    double start, end;
    double time_std, time_custom;

    start = get_time_sec();
    for (size_t i = 0; i < iterations; ++i) {
        memmove(dest_std, src, size);
    }
    end = get_time_sec();
    time_std = end - start;

    start = get_time_sec();
    for (size_t i = 0; i < iterations; ++i) {
        _memmove(dest_custom, src, size);
    }
    end = get_time_sec();
    time_custom = end - start;

    if (memcmp(dest_std, dest_custom, size) != 0) {
        fprintf(stderr, "Erreur: Les résultats de memmove et _memmove diffèrent pour la taille %zu\n", size);
    } else {
        printf("Taille: %8zu bytes | memmove: %8.6f s | _memmove: %8.6f s | Gain: %.2f%%\n",
               size, time_std, time_custom,
               ((time_std - time_custom) / time_std) * 100);
    }

    // Libérer la mémoire
    free(src);
    free(dest_std);
    free(dest_custom);
}

int main() {
    srand(time(NULL));

    size_t sizes[] = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    size_t iterations = 10000;

    printf("Benchmark de memmove vs _memmove\n");
    printf("Iterations: %zu\n\n", iterations);
    printf("Taille (bytes) | memmove (s) | _memmove (s) | Gain\n");
    printf("---------------------------------------------------------\n");

    for (size_t i = 0; i < num_sizes; ++i) {
        benchmark_memmove(sizes[i], iterations);
    }

    return 0;
}
