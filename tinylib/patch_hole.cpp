#include <cstdint>
#include <cstdio>
#include <vector>

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

enum CantPatchReason
{
	NotEnoughCoordinates = -1,
	NoClosestPointOnSkeleton = -2,
	No2DisjointRegions = -3,
	NoClosestPointOnLines = -4
};

extern "C" void skeletonize(const uint8_t* image, int im_stride, uint8_t* skeleton, int sk_stride, int width, int height);

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
extern "C" int patch_hole(uint8_t* lines, int lines_stride, uint8_t* skeleton, int sk_stride, int width, int height, int cx, int cy,
	       int patch_region_w, int patch_region_h, int* xs, int* ys, int max_coord,
	       int* xs1, int* ys1, int *max1, int* xs2, int* ys2, int *max2)
{
	if(max_coord < 3) {
		return NotEnoughCoordinates;
	}
	int closest_x, closest_y;
	bool found = find_closest_point(width, height, cx, cy, closest_x, closest_y, [&](int x, int y) { return skeleton[sk_stride*y + x]; });
	if(!found) {
		return NoClosestPointOnSkeleton;
	}
	const int region_color[2] = {2,3};
	int pass = 0;
	while(pass < 2 && patch_region_w > 1 && patch_region_h > 1) {
		int xstart = fixed_size_region_1d(closest_x, patch_region_w, width);
		int ystart = fixed_size_region_1d(closest_y, patch_region_h, height);
		uint8_t* patch = skeleton + sk_stride*ystart + xstart;

		//find a neighbor point which is not a part of the skeleton and flood-fill it with a first color,
		//and then another which is not a part of the skeleton nor was flooded by the first flood-fill,
		//and flood-fill it with a second color
		//
		//make the patch iteratively smaller. big patches are good because you can patch big holes,
		//but if a patch is too big, you might not see two disjoint colors. smaller patches are more
		//likely to give you two disjoint colors, but can only patch smaller holes. if we fail to find
		//disjoint colors at the largest patch size, try smaller ones before giving up
		int flood_x = 0, flood_y = 0;
		for(int yo=-1; yo<=1 && pass < 2; ++yo) {
			for(int xo=-1; xo<=1 && pass < 2; ++xo) {
				int x = closest_x + xo;
				int y = closest_y + yo;
				if(x < 0 || x >= width || y < 0 || y >= height) {
					continue;
				}
				if(!skeleton[sk_stride*y + x]) {
					flood_x = x-xstart;
					flood_y = y-ystart;
					flood_fill_mask(patch, sk_stride, patch_region_w, patch_region_h, flood_x, flood_y, region_color[pass], nullptr, 0);
					pass++;
				}
			}
		}
		if(pass < 2) {
			//clear the are we flooded during the previous attempt
			flood_fill_mask(patch, sk_stride, patch_region_w, patch_region_h, flood_x, flood_y, 0, nullptr, 0);
			//TODO: do a better search than this?..
			patch_region_w -= 5;
			patch_region_h -= 5;

			pass = 0;
		}
	}
	if(pass < 2) {
		return No2DisjointRegions; //couldn't find 2 disjoint regions
	}

	xs[1] = closest_x;
	ys[1] = closest_y;
	for(int c=0; c<2; ++c) {
		found = find_closest_point(width, height, cx, cy, xs[c*2], ys[c*2], [&](int x, int y) {
			return lines[lines_stride*y + x] == 255 && skeleton[sk_stride*y + x] == region_color[c];
		});
		if(!found) {
			return NoClosestPointOnLines;
		}
	}

	//skeletonize the lines and try to find additional points for line fitting so we have a smooth fit
	//(this helps when patching a hole in a circle, for example; doesn't work when patching a hole between
	//two nearly parallel lines - in the latter case we will hit a "junction" in the skeleton immediately,
	//finding nothing
	for(int y=0; y<height; ++y) {
		for(int x=0; x<width; ++x) {
			int i = lines_stride*y + x;
			lines[i] = lines[i] == 255; //skeletonize expects a binary image
		}
	}

	std::vector<uint8_t> lines_skeleton(width*height);
	skeletonize(lines, lines_stride, &lines_skeleton[0], width, width, height);
	//fill the boundary with 1s if we had 255 there (special case of patching holes near image boundaries)
	for(int y=0; y<height; ++y) {
		for(int x=0; x<width; ++x) {
			if(x==0 || y==0 || x==width-1 || y==height-1) {
				int i = lines_stride*y + x;
				lines[i] = lines[i] == 255;
			}
		}
	}

	int* xarr[] = {xs1, xs2};
	int* yarr[] = {ys1, ys2};
	int* maxarr[] = {max1, max2};
	for(int c=0; c<2; ++c) {
		int cx, cy;
		//we can't assume that the closest points are on the lines _skeleton_ - they're definitely
		//on the lines but not necessarily on the skeleton which is "thinner"
		found = find_closest_point(width, height, xs[c*2], ys[c*2], cx, cy, [&](int x, int y) {
			return lines_skeleton[width*y + x];
		});
		//printf("closest on lines was %d %d -> on the skeleton %d %d\n", xs[c*2], ys[c*2], cx, cy);
		if(!found) {
			*maxarr[c] = 0;
			continue;
		}
		int i;
		int maxc = *maxarr[c];
		int px = cx, py = cy; //remember the previous non-zero neighbor to not traverse it again
		for(i=0; i<maxc; ++i) {
			//if we have a single non-zero neighbor, add it to the coordinate array
			int nzx = 0;
			int nzy = 0;
			int num_nz = 0;
			for(int yo=-1; yo<=1; ++yo) {
				for(int xo=-1; xo<=1; ++xo) {
					int x = cx + xo;
					int y = cy + yo;
					if(x < 0 || x >= width || y < 0 || y >= height) {
						continue;
					}
					if((x == cx && y == cy) || (x == px && y == py)) {
						continue;
					}
					if(lines_skeleton[width*y + x]) {
						//printf("nz: %d %d - %d %d \n", x, y, yo, xo);
						num_nz++;
						nzx = x;
						nzy = y;
					}
					else {
						//printf("z: %d %d - ls %d color %d [%d]\n", x, y, lines_skeleton[width*y + x], skeleton[sk_stride*y + x] == region_color[c], skeleton[sk_stride*y + x]);
					}
				}
			}
			if(num_nz == 1) {
				xarr[c][i] = nzx;
				yarr[c][i] = nzy;
				px = cx;
				py = cy;
				cx = nzx;
				cy = nzy;
			}
			else {  //if we have no non-zero neighbors, we obviously can't add any more
			        //coordinates for line fitting; if we have more than one neighbor,
			        //it's a "fork" and we can't guess which path to take
			        //printf("c=%d quitting at i=%d x=%d y=%d found %d non-zero neighbors\n", c, i, cx, cy, num_nz);
				break;
			}
		}
		*maxarr[c] = i;
	}
	//for debugging
	//for(int y=0; y<height; ++y) {
	//	for(int x=0; x<width; ++x) {
	//		skeleton[sk_stride*y + x] = lines_skeleton[width*y + x];
	//	}
	//}

	return 3;
}
