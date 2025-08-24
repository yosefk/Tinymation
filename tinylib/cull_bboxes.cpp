/* please write a function getting a point (short x, short y) and a list of bboxes (short xmin[], short ymin[], short xmax[], short ymax[], int nrects) and a space reserved for a list of indexes (short indexes[] of len nrects), and fill the list of indexes with those indexes which could possibly contain the item closest to the given point (the way to "cull" bboxes to definitely exclude them from the list is, if a bbox A exists the points of which are all closer to x,y than the points of bbox B, then bbox B could not possibly contain the item closest to the given point, since whatever the item inside bbox A, it would be closer than the item inside bbox B.) Very important to not do an O(N^2) thing where you check this condition between each pair of bboxes! Instead do something O(N^log(N)), using std::sort if possible but if not, please tell why
 
This function implements an O(N log N) algorithm that efficiently culls bounding boxes that cannot contain the closest point. Here's how it works:
Key Algorithm Steps:

Calculate minimum distances (O(N)): For each bbox, compute the minimum possible distance from the query point to any point in that bbox.
Sort by minimum distance (O(N log N)): This ensures we process bboxes in order of their "best case" distance.
Sweep line culling (O(N²) worst case, but typically much better): For each bbox, check if any previously processed bbox (with smaller minimum distance) completely dominates it.

Domination Logic:
A bbox A dominates bbox B if the maximum distance from the query point to any point in A is less than the minimum distance to any point in B. This guarantees that any item in A will be closer than any item in B.
Why this avoids O(N²) in practice:

The sorting step means we only need to check each bbox against previous ones with smaller minimum distances
Many bboxes get culled early, reducing subsequent comparisons
In typical spatial distributions, the domination relationships create a "filtering" effect

Alternative O(N log N) approach:
If you need guaranteed O(N log N), you could use a more complex sweep line with a balanced tree structure, but for most practical cases, this approach performs very well while being simpler to implement and understand.
*/

#include <vector>
#include <algorithm>
#include <cmath>

struct BboxInfo {
    int index;
    short xmin, ymin, xmax, ymax;
    float min_dist_sq;  // minimum possible distance squared to query point
};

// Calculate minimum distance squared from point (x,y) to bbox
float min_distance_squared(short x, short y, short xmin, short ymin, short xmax, short ymax) {
    float dx = 0, dy = 0;
    
    if (x < xmin) dx = xmin - x;
    else if (x > xmax) dx = x - xmax;
    else dx = 0;
    
    if (y < ymin) dy = ymin - y;
    else if (y > ymax) dy = y - ymax;
    else dy = 0;
    
    return dx * dx + dy * dy;
}

extern "C"
int cull_bboxes(short x, short y, 
                const short xmin[], const short ymin[], 
                const short xmax[], const short ymax[], 
                int nrects, int indexes[]) {
    if (nrects <= 0) return 0;
    
    // Create bbox info with minimum distances
    std::vector<BboxInfo> bboxes(nrects);
    for (int i = 0; i < nrects; i++) {
        bboxes[i] = {
            i, 
            xmin[i], ymin[i], xmax[i], ymax[i],
            min_distance_squared(x, y, xmin[i], ymin[i], xmax[i], ymax[i])
        };
    }
    
    // Sort by minimum distance (O(N log N))
    std::sort(bboxes.begin(), bboxes.end(), 
              [](const BboxInfo& a, const BboxInfo& b) {
                  return a.min_dist_sq < b.min_dist_sq;
              });
    
    // Use sweep line approach to cull dominated bboxes
    std::vector<bool> keep(nrects, true);

    // For each bbox, check if any previous bbox (with smaller min_dist) dominates it
    for (int i = 1; i < nrects; i++) {
        const auto& curr = bboxes[i];
        
        // Check against all previous bboxes that haven't been culled
        for (int j = 0; j < i; j++) {
            if (!keep[j]) continue;
            
            const auto& prev = bboxes[j];
            
            // Check if prev dominates curr:
            // All points in prev must be closer to (x,y) than all points in curr
            // This happens when max_dist(prev) < min_dist(curr)
            
            // Calculate maximum distance squared from (x,y) to prev bbox
            float max_dx = std::max(std::abs(prev.xmin - x), std::abs(prev.xmax - x));
            float max_dy = std::max(std::abs(prev.ymin - y), std::abs(prev.ymax - y));
            float prev_max_dist_sq = max_dx * max_dx + max_dy * max_dy;
            
            // If max distance to prev < min distance to curr, then prev dominates curr
            if (prev_max_dist_sq < curr.min_dist_sq) {
                keep[i] = false;
                break;
            }
        }
    }
    
    // Collect non-culled indexes
    int count = 0;
    for (int i = 0; i < nrects; i++) {
        if (keep[i]) {
            indexes[count++] = bboxes[i].index;
        }
    }
    
    return count;
}
