#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

class Barrier {
private:
    std::mutex mtx;
    std::condition_variable cv;
    int count = 0;
    int initial_count;

public:
    explicit Barrier(int count) : initial_count(count), count(count) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        
        if (--count == 0) {
            count = initial_count;
            cv.notify_all();
        } else {
            std::cout << "Count: " << count << "\n";
            // Wait until all threads reach the barrier
            // and reset the count for the next round
            cv.wait(lock, [this] { return count == initial_count; });
        }
    }
};

void worker(Barrier& barrier) {
    std::cout << "Thread " << std::this_thread::get_id() << " is doing work...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1)); // Simulate work
    std::cout << "Thread " << std::this_thread::get_id() << " is waiting at the barrier...\n";
    barrier.wait();
    std::cout << "Thread " << std::this_thread::get_id() << " has passed the barrier!\n";
}

int main() {
    const int num_threads = 5;
    Barrier barrier(num_threads);

    std::thread threads[num_threads];
    for (int i = 0; i < num_threads; ++i) {
        threads[i] = std::thread(worker, std::ref(barrier));
    }

    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
    }

    return 0;
}