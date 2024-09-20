//based on https://github.com/scikit-image/scikit-image/blob/v0.23.2/skimage/morphology/_skeletonize_cy.pyx
//
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <algorithm>

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

//note that this algorithm parallelizes decently over rows, but ATM tinylib doesn't use any parallelism
extern "C" void skeletonize(const uint8_t* image, int im_stride, uint8_t* skeleton, int sk_stride, int width, int height) 
{
    width += 2;
    height += 2;
	uint8_t* curr_skeleton = new uint8_t[width*height];
	uint8_t* next_skeleton = new uint8_t[width*height]; 
    memset(curr_skeleton, 0, width*height);
    memset(next_skeleton, 0, width*height);

    for(int y=1; y<height-1; ++y) {
        for(int x=1; x<width-1; ++x) {
            int oft = y*width + x;
            int pix = image[(y-1)*im_stride + x-1];
            curr_skeleton[oft] = pix;
            next_skeleton[oft] = pix;
        }
    }

	int passes = 0;

	int l = -1;
	int r = +1;
	int d = -width;
	int u = +width;

	bool pixel_removed = true;

	while(pixel_removed) {
		pixel_removed = false;
		for(int pass_num=1; pass_num<3; ++pass_num) {
//			int nonzero = 0;
			passes++;

			for(int y=1; y<height-1; ++y) {
				uint8_t* curr_row = curr_skeleton + width*y;
				uint8_t* next_row = next_skeleton + width*y;
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
							pixel_removed = true;
						}
					}
				}
			}
			memcpy(curr_skeleton, next_skeleton, width * height);
//			printf("nonzero: %d\n", nonzero);
		}
	}
    //copy the skeleton to the output buffer
    for(int y=1; y<height-1; ++y) {
        for(int x=1; x<width-1; ++x) {
            skeleton[sk_stride*(y-1) + x-1] = curr_skeleton[width*y + x];
        }
    }
    delete [] curr_skeleton;
	delete [] next_skeleton;
	//printf("total passes: %d\n", passes);
}
