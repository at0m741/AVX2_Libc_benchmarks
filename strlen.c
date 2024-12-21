#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline size_t _strlen_avx(const char *str)
{
    if (!str) return 0;

    const char *start = str;
    __m256i zero = _mm256_set1_epi8(0);
    uintptr_t addr = (uintptr_t)str;
    size_t mis = addr & 31;

    if (mis != 0)
    {
        size_t partial = 32 - mis; 
        __m256i data = _mm256_loadu_si256((const __m256i*)str);
        __m256i cmp  = _mm256_cmpeq_epi8(zero, data);
        int mask     = _mm256_movemask_epi8(cmp);

        if (mask != 0)
        {
            int idx = __builtin_ctz(mask);
            return (size_t)(str + idx - start);
        }
        str += partial;  
    }

    for (;;)
    {
        __m256i data = _mm256_load_si256((const __m256i*)str);
        __m256i cmp  = _mm256_cmpeq_epi8(zero, data);
        int mask     = _mm256_movemask_epi8(cmp);

        if (mask != 0)
        {
            int idx = __builtin_ctz(mask);
            return (size_t)(str + idx - start);
        }
        str += 32;
    }
    return 0;
}

double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

size_t round_up_to_multiple(size_t num, size_t multiple) {
    if (multiple == 0)
        return num;
    size_t remainder = num % multiple;
    if (remainder == 0)
        return num;
    return num + multiple - remainder;
}

void benchmark_strlen(const char *str, size_t size, size_t iterations, size_t *accum_std, size_t *accum_custom) {
    size_t len_std = 0;
    size_t len_custom = 0;

    double start = get_time_sec();
    for (size_t i = 0; i < iterations; ++i) {
        len_std = strlen(str);
        *accum_std += len_std;
    }
    double end = get_time_sec();
    double time_std = end - start;
    start = get_time_sec();
    for (size_t i = 0; i < iterations; ++i) {
        len_custom = _strlen_avx(str);
        *accum_custom += len_custom;
    }
    end = get_time_sec();
    double time_custom = end - start;

    if (len_std != len_custom) {
        fprintf(stderr, "Error: strlen and _strlen_avx return differents sizes (%zu vs %zu)  %zu\n", len_std, len_custom, size);
    } else {
        printf("size: %10zu bytes | strlen: %8.6f s | _strlen_avx: %8.6f s | Gain: %.2f%%\n",
               size, time_std, time_custom,
               ((time_std - time_custom) / time_std) * 100);
    }
}

int main() {
    srand((unsigned int)time(NULL));
    size_t sizes[] = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    size_t iterations = 200000;

    printf("Benchmark strlen vs _strlen_avx\n");
    printf("Iters: %zu\n\n", iterations);
    printf("size (bytes) | strlen (s) | _strlen_avx (s) | Gain\n");
    printf("---------------------------------------------------------\n");

    size_t accum_std_total = 0;
    size_t accum_custom_total = 0;

    for (size_t i = 0; i < num_sizes; ++i) {
        size_t size = sizes[i];
        size_t aligned_size = round_up_to_multiple(size + 1, 32); 

        char *str = (char *)aligned_alloc(32, aligned_size);
        if (!str) {
            fprintf(stderr, "Alloc error for = %zu\n", size);
            exit(EXIT_FAILURE);
        }

        for (size_t j = 0; j < size; ++j) {
            str[j] = 'A' + (rand() % 26); 
        }
        str[size] = '\0'; 

        benchmark_strlen(str, size, iterations, &accum_std_total, &accum_custom_total);

        free(str);
    }

    if (accum_std_total == 0 || accum_custom_total == 0) {
        printf("Accumulation: std=%zu, custom=%zu\n", accum_std_total, accum_custom_total);
    }

    return 0;
}
