#include <immintrin.h>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;

int *a, *b, *c, *k;
constexpr int N = 100;

int main() {
    a = new int[N];
    b = new int[N];
    c = new int[N];
    k = new int[N];

    for(int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i * 1000;
        k[i] = 10;
        c[i] = 0;
    }

    #pragma omp parallel for
    for (long long int i = 0; i < N; i += 8) {
        __m256i A = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i B = _mm256_loadu_si256((__m256i*)&b[i]);
        __m256i K = _mm256_loadu_si256((__m256i*)&k[i]);
        __m256i C_1 = _mm256_mullo_epi32(A, K);
        __m256i C_2 = _mm256_mullo_epi32(B, K);
        __m256i C = _mm256_add_epi32(C_1, C_2);

        _mm256_storeu_si256((__m256i*)&c[i], C);
    }

}