#include <cstdint>
#include <cstdio>

extern "C" void flood_fill_mask(unsigned char* mask, int mask_stride,
	       int width, int height, int seed_x, int seed_y, int mask_new_val,
	       int* region, int _8_connectivity);

template<class Pred>
bool find_closest_point(int width, int height, int cx, int cy, int& closest_x, int& closest_y, const Pred& pred)
{
	closest_x = -1;
	closest_y = -1;
	float min_dist_squared = 1000000;
	bool found = false;
	for(int y=0; y<height; ++y) {
		for(int x=0; x<width; ++x) {
			if(pred(x,y)) {
				float dist_squared = (cx-x)*(cx-x) + (cy-y)*(cy-y);
				if(dist_squared < min_dist_squared) {
					closest_x = x;
					closest_y = y;
					min_dist_squared = dist_squared;
					found = true;
				}
			}
		}
	}
	return found;
}

int fixed_size_region_1d(int center, int part, int full)
{
    if(center < part/2) {
        return 0;
    }
    else if(center > full - part/2) {
        return full - part;
    }
    else {
        return center - part/2;
    }
}

//we modify the skeleton in-place
//
//find the closest point to x,y on the skeleton
//find 2 points on opposite sides of the skeleton
//flood-fill the skeleton image from both points; can only patch
//  if these are two distinct regions - check if the second flooding
//  overwrote the first
//find the closest point to x,y in the lines image within each flooded region
//  these are the two points we want to connect with a line to patch the hole
//
//writes coordinates of the points to draw a line through into xs[] and ys[]
//returns a non-zero number of coordinates iff there's a hole to patch
//and there was enough room in xs[] and ys[] for the output
int patch_hole(const uint8_t* lines, int lines_stride, uint8_t* skeleton, int sk_stride, int width, int height, int cx, int cy,
	       int patch_region_w, int patch_region_h, int* xs, int* ys, int max_coord)
{
	if(max_coord < 3) {
		return 0;
	}
	int closest_x, closest_y;
	bool found = find_closest_point(width, height, cx, cy, closest_x, closest_y, [&](int x, int y) { return skeleton[sk_stride*y + x]; });
	if(!found) {
		return 0;
	}
	int xstart = fixed_size_region_1d(closest_x, patch_region_w, width);
	int ystart = fixed_size_region_1d(closest_y, patch_region_h, height);
	uint8_t* patch = skeleton + sk_stride*ystart + xstart;

	//find a neighbor point which is not a part of the skeleton and flood-fill it with a first color,
	//and then another which is not a part of the skeleton nor was flooded by the first flood-fill,
	//and flood-fill it with a second color
	const int region_color[2] = {2,3};
	int pass = 0;
	for(int yo=-1; yo<=1 && pass < 2; ++yo) {
		for(int xo=-1; xo<=1 && pass < 2; ++xo) {
			int x = closest_x + xo;
			int y = closest_y + yo;
			if(x < 0 || x >= width || y < 0 || y >= height) {
				continue;
			}
			if(!skeleton[sk_stride*y + x]) {
				flood_fill_mask(patch, sk_stride, patch_region_w, patch_region_h, x-xstart, y-ystart, region_color[pass], nullptr, 0);
				pass++;
			}

		}
	}
	if(pass < 2) {
		return 0; //couldn't find 2 disjoint regions
	}

	xs[1] = closest_x;
	ys[1] = closest_y;
	for(int c=0; c<2; ++c) {
		found = find_closest_point(width, height, cx, cy, xs[c*2], ys[c*2], [&](int x, int y) {
			return lines[lines_stride*y + x] == 255 && skeleton[sk_stride*y + x] == region_color[c];
		});
		if(!found) {
			return 0;
		}
	}

	return 3;
}
