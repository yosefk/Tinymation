#include <tbb/tbb.h>

static tbb::global_control* gc = new tbb::global_control(tbb::global_control::max_allowed_parallelism, 4);

// Set the number of TBB worker threads
extern "C" void parallel_set_num_threads(int num_threads) {
    delete gc;
    gc = new tbb::global_control(tbb::global_control::max_allowed_parallelism, num_threads);
}

extern "C" void parallel_for_grain(void (*range_func)(int, int), int start, int finish, int grain_size)
{
    tbb::parallel_for(tbb::blocked_range<int>(start, finish, grain_size), [&](const tbb::blocked_range<int>& r) {
	range_func(r.begin(), r.end());
    });
}
