#include "thread_pool.h"

ThreadPool::ThreadPool(int num_workers)
{
    _start = new std::atomic<bool>(num_workers-1);
    for(int i=0; i<num_workers-1; ++i) {
        _start[i] = false;
        _workers.emplace_back([i,this] { worker_entry_point(i); });
    }
}

void ThreadPool::parallel_for(int n, const std::function<void(int)>& func)
{
    _item.func = &func;
    _item.next_ind = 0;
    _item.n = n;
    _item.to_do = n;

    for(int i=0; i<(int)_workers.size(); ++i) {
        _start[i] = true;
    }

    work(_workers.size());

    //wait until the work is done (we might be out of work but other workers
    //might be finishing the last tasks)
    while(_item.to_do > 0);
}

void ThreadPool::worker_entry_point(int i)
{
    auto& start = _start[i];
    while(true) {
        if(start) {
            start = false;
	    work(i);
	}
    }
}

void ThreadPool::work(int i)
{
    int n = _item.n;
    while(_item.next_ind < n) {
        int next_ind = _item.next_ind++;
	//fprintf(stderr,"worker %d got %d\n", i, next_ind);
        if(next_ind < n) { /* it could have exceeded n because of the concurrent increment above */
            (*_item.func)(next_ind);
            --_item.to_do; /* once this reaches 0, nobody needs this work item */
        }
    }
}
