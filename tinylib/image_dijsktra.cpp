#include <cstdint>
#include <algorithm>
#include <queue>

const float NO_PATH_DIST = 1000000; //output value for coordinates unreachable via non-zero points from the non-zero point closest to cx, cy

struct IDPoint
{
	float dist;
	int16_t x;
	int16_t y;

	bool operator<(const IDPoint& p) const { return dist < p.dist; }
};

//returns the maximum path length from the non-zero point closest to (cx,cy) or NO_PATH_DIST if there's no such point
int image_dijkstra(const uint8_t* image, int im_stride, float* dist, int d_stride, int width, int height, int cx, int cy)
{
	std::fill(dist, dist+d_stride*height, NO_PATH_DIST);
	std::vector<bool> enqueued(width*height);
	std::fill(enqueued.begin(), enqueued.end(), false);

	//find the closest non-zero point in the image to (cx,cy)
	int closest_x = -1, closest_y = -1;
	float min_dist_squared = NO_PATH_DIST;
	for(int y=0; y<height; ++y) {
		for(int x=0; x<width; ++x) {
			if(image[im_stride*y + x]) {
				float dist_squared = (cx-x)*(cx-x) + (cy-y)*(cy-y);
				if(dist_squared < min_dist_squared) {
					closest_x = x;
					closest_y = y;
					min_dist_squared = dist_squared;
				}
			}
		}
	}
	if(min_dist_squared == NO_PATH_DIST) {
		return NO_PATH_DIST; //no pixels set
	}

	std::priority_queue<IDPoint, std::vector<IDPoint>> q;
	auto enqueue = [&](float distance, int x, int y) {
		q.push({distance, int16_t(x), int16_t(y)});
		dist[d_stride*y + x] = distance;
		enqueued[width*y + x] = true;
	};
	enqueue(0, closest_x, closest_y);

	int iter = 0;
	while(!q.empty()) {
		++iter;
		IDPoint p = q.top();
		q.pop();
		enqueued[width*p.y + p.x] = false;

		for(int yo=-1; yo<=1; ++yo) {
			for(int xo=-1; xo<=1; ++xo) {
				int x = p.x + xo;
				int y = p.y + yo;
				if(x < 0 || x >= width || y < 0 || y >= height) {
					continue;
				}
				if(!image[im_stride*y + x]) {
					continue;
				}
				float extra_dist = (xo==0 || yo==0) ? 1 : 1.44; //sqrt(2)
				float neigh_dist = p.dist + extra_dist;
				if(neigh_dist < dist[d_stride*y + x]) {
					dist[d_stride*y + x] = neigh_dist;
					if(!enqueued[d_stride*y + x]) {
						enqueue(neigh_dist, x, y);
					}
				}

			}
		}
	}
	float max_dist = 0;
	for(int i=0; i<width*d_stride; ++i) {
		if(dist[i] != NO_PATH_DIST) {
			max_dist = std::max(max_dist, dist[i]);
		}
	}
	return max_dist;
}
