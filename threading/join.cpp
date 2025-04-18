#include <iostream>
#include <thread>
#include <vector>

using namespace std;

int *a, *b, *c, *k;
constexpr int N = 100;
constexpr int NT = 10;

void mac(int tid, int num_threads) {
    int start = tid * (N / num_threads);
    int end = (tid + 1) * (N / num_threads);

    for(int i = start; i < end; ++i) {
        c[i] = a[i] * k[i];
        c[i] += b[i] * k[i];

    }
}

int main(int argc, char* argv[]) {

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

    vector<thread> threads;
    for (int i = 0; i < NT; ++i) {
        threads.emplace_back(mac, i, NT);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    for (int i = 0; i < N; ++i) {
        cout << "c[" << i << "] = " << c[i] << endl;
    }
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] k;

    return 0;
}