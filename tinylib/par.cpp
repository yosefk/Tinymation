const int default_workers = 4;
static int num_workers = default_workers;

#define PAR_TBB

#ifdef PAR_TBB

#include <tbb/tbb.h>

static tbb::global_control* gc = new tbb::global_control(tbb::global_control::max_allowed_parallelism, default_workers);

// Set the number of TBB worker threads
extern "C" void parallel_set_num_threads(int num_threads) {
    delete gc;
    gc = new tbb::global_control(tbb::global_control::max_allowed_parallelism, num_threads);
    num_workers = num_threads;
}

extern "C" void parallel_for_grain(void (*range_func)(int, int), int start, int finish, int grain_size)
{
    if(finish <= start) {
        return;
    }
    if(grain_size == 0) {
        grain_size = (finish - start + num_workers - 1) / num_workers;
    }
    tbb::parallel_for(tbb::blocked_range<int>(start, finish, grain_size), [=](const tbb::blocked_range<int>& r) {
        range_func(r.begin(), r.end());
    });
}

#else 

#include "thread_pool.h"

static ThreadPool* pool = nullptr;

extern "C" void parallel_set_num_threads(int num_threads)
{
    if(!pool) {
        pool = new ThreadPool(num_threads);
    }
}

extern "C" void parallel_for_grain(void (*range_func)(int, int), int start, int finish, int grain_size)
{
    if(finish <= start) {
        return;
    }
    if(grain_size == 0) {
        grain_size = (finish - start + (num_workers*4) - 1) / (num_workers*4);
        //grain_size = 1;
    }
    //fprintf(stderr,"starting parallel_for %d %d %d\n", start, finish, grain_size);
    pool->parallel_for((finish - start + grain_size - 1) / grain_size, [=](int item) {
        int begin = start + item*grain_size;
        int end = std::min(begin + grain_size, finish);
        range_func(begin, end);
    });
}

#endif
