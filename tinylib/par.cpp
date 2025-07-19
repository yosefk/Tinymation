#include <tbb/tbb.h>

const int default_workers = 4;
static int num_workers = default_workers;
static tbb::global_control* gc = new tbb::global_control(tbb::global_control::max_allowed_parallelism, default_workers);

// Set the number of TBB worker threads
extern "C" void parallel_set_num_threads(int num_threads) {
    delete gc;
    gc = new tbb::global_control(tbb::global_control::max_allowed_parallelism, num_threads);
    num_workers = num_threads;
}

extern "C" void parallel_for_grain(void (*range_func)(int, int), int start, int finish, int grain_size)
{
    if(grain_size == 0) {
	grain_size = (finish - start + num_workers - 1) / num_workers;
    }
    tbb::parallel_for(tbb::blocked_range<int>(start, finish, grain_size), [&](const tbb::blocked_range<int>& r) {
	range_func(r.begin(), r.end());
    });
}
