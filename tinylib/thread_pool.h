#pragma once

#include <vector>
#include <thread>
#include <functional>
#include <atomic>

class ThreadPool
{
  public:
    ThreadPool(int num_workers); //creates num_workers-1 threads
    void busy_wait_for_work(); //prepare for work to arrive - workers enter busy wait loop
    void rest(); //exit 
    void parallel_for(int n, const std::function<void(int)>& func);

  private:
    struct WorkItem
    {
        const std::function<void(int)>* func;
	std::atomic<int> next_ind;
	std::atomic<int> to_do;
	int n;
       
    };
    void work(int i);
    void worker_entry_point(int i);

    std::vector<std::thread> _workers;
    std::atomic<bool>* _start;
    WorkItem _item;
};
