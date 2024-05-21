//based on https://github.com/scikit-image/scikit-image/blob/v0.23.2/skimage/morphology/_skeletonize_cy.pyx
//
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>

static const uint8_t lut[256] =
                            {0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0,
                             1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0,
                             3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             2, 0, 0, 0, 3, 0, 2, 2, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
                             0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0,
                             3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0,
                             2, 0, 0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0,
                             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3,
                             0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                             0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0,
                             2, 0, 0, 0};

static int threads = atoi(getenv("TBB_NUM_THREADS") ? getenv("TBB_NUM_THREADS") : "4");

//TODO: resurrect; ATM don't want to bother with TBB, margins...
#if SKEL
void skeletonize(const uint8_t* image, int im_stride, uint8_t* skeleton, int sk_stride, int width, int height) 
{
	int l = -1;
	int r = +1;
	int d = -sk_stride;
	int u = +sk_stride;

	bool pixel_removed = true;

	uint8_t* curr_skeleton = skeleton;
	uint8_t* next_skeleton = new uint8_t[sk_stride*height]; 

	//FIXME: do this properly...
	uint8_t* fimg = (uint8_t*)image;
	memset(fimg, 0, im_stride);
	memset(fimg+im_stride*(height-1), 0, im_stride);
	for(int y=0; y<height; ++y) {
		uint8_t* r = fimg + im_stride*y;
		r[0]=0;
		r[width-1]=0;
	}

	//FIXME: copy per row!! we have different strides... or force them being the same...
	memcpy(curr_skeleton, image, sk_stride * height);
	memcpy(next_skeleton, image, sk_stride * height);

//#pragma omp parallel
	struct Removed
	{
		bool was=false;
		unsigned char pad[63];
	};

	Removed* removed = new Removed[height];
	Removed* premoved = new Removed[height];
	Removed was;
	was.was = true;
	std::fill(removed, removed+height, was);

	int passes = 0;

	printf("max_allowed_parallelism %d\n", threads);
	oneapi::tbb::global_control global_limit(oneapi::tbb::global_control::max_allowed_parallelism, threads);

	int skipped = 0;
	int total = 0;

	while(pixel_removed) {
		pixel_removed = false;
		std::copy(removed, removed+height, premoved);
		std::fill(removed, removed+height, Removed());
		for(int pass_num=1; pass_num<3; ++pass_num) {
//			int nonzero = 0;
			passes++;
//#pragma omp parallel for num_threads(16) schedule(dynamic, 16)


			tbb::parallel_for(tbb::blocked_range<int>(1, height-1, 16), [&](const tbb::blocked_range<int>& ys) {
#if 0
			    bool passive = true;
			    for(int y=ys.begin()-1; y<ys.end()+1; ++y) {
			    	if(premoved[y].was) {
					passive = false;
					break;
				}
			    }
			    total++;
			    if(passive) {
			    	    skipped++;
				    return;
			    }
#endif
			    for(int y=ys.begin(); y<ys.end(); ++y) {
//
//			for(int y=1; y<height-1; ++y) {
				uint8_t* curr_row = curr_skeleton + sk_stride*y;
				uint8_t* next_row = next_skeleton + sk_stride*y;
				for(int x=1; x<width-1; ++x) {
					const uint8_t* m = curr_row + x;
					if(*m) {
//						nonzero++;
						int neighbors = lut[(m[d+l] << 0) |
							            (m[d] << 1) |
								    (m[d+r] << 2) |
								    (m[r] << 3) |
								    (m[u+r] << 4) |
								    (m[u] << 5) |
								    (m[u+l] << 6) |
								    (m[l] << 7)];

						if(neighbors == 3 || neighbors == pass_num) {
							next_row[x] = 0;
							//pixel_removed = true;
							removed[y].was = true;
						}
					}
				}
//			}
			    }
			}, tbb::simple_partitioner());
			//
			//
			memcpy(curr_skeleton, next_skeleton, sk_stride * height);
//			printf("nonzero: %d\n", nonzero);
		}
		for(int y=0; y<height; ++y) {
			if(removed[y].was) {
				pixel_removed = true;
				break;
			}
		}
	}
	delete [] next_skeleton;
	printf("total passes: %d\n", passes);
	printf("skipped %f\n", 100*double(skipped)/total);
}
#endif
