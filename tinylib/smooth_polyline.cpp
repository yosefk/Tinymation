#define _USE_MATH_DEFINES
#include <vector>
#include <cmath>
#include <algorithm>

double euclidean_distance(double x1, double y1, double x2, double y2) {
    return std::sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

template<typename F>
void compute_polyline_distances(std::vector<double>& polyline_dists, int npoints, F&& segment_length, int closest_idx, int closed) {
    polyline_dists.resize(npoints);
    polyline_dists[closest_idx] = 0;
    
    if (closed) {
        // For closed polylines, compute distances in both directions and take minimum
        std::vector<double> forward_dists(npoints, std::numeric_limits<double>::infinity());
        std::vector<double> backward_dists(npoints, std::numeric_limits<double>::infinity());
        
        forward_dists[closest_idx] = 0;
        backward_dists[closest_idx] = 0;
        
        // Forward direction
        for (int i = 1; i < npoints; ++i) {
            int curr_idx = (closest_idx + i) % npoints;
            int prev_idx = (closest_idx + i - 1 + npoints) % npoints;
            forward_dists[curr_idx] = forward_dists[prev_idx] + segment_length(prev_idx, curr_idx);
        }
        
        // Backward direction  
        for (int i = 1; i < npoints; ++i) {
            int curr_idx = (closest_idx - i + npoints) % npoints;
            int next_idx = (closest_idx - i + 1 + npoints) % npoints;
            backward_dists[curr_idx] = backward_dists[next_idx] + segment_length(curr_idx, next_idx);
        }
        
        // Take minimum of forward and backward distances
        for (int i = 0; i < npoints; ++i) {
            polyline_dists[i] = std::min(forward_dists[i], backward_dists[i]);
        }
    } else {
        // For open polylines, use the original approach
        // Forward direction from closest point
        for (int i = closest_idx + 1; i < npoints; ++i) {
            polyline_dists[i] = polyline_dists[i-1] + segment_length(i-1, i);
        }
        // Backward direction from closest point
        for (int i = closest_idx - 1; i >= 0; --i) {
            polyline_dists[i] = polyline_dists[i+1] + segment_length(i, i+1);
        }
    }
}

template<typename F>
void compute_distances_to_closest_corner(std::vector<double>& dist_to_closest_corner, int npoints, F&& segment_length, int closed, const unsigned char* is_corner) {
    dist_to_closest_corner.resize(npoints);
    std::fill(dist_to_closest_corner.begin(), dist_to_closest_corner.end(), std::numeric_limits<double>::infinity());
    
    if (is_corner == nullptr) {
        return;
    }
    
    if (closed) {
        //TODO: make this faster by only considering the 2 corners closest to each point,
        //instead of considering every corner from every point
        // Find all corner positions first
        std::vector<int> corner_positions;
        for (int i = 0; i < npoints; ++i) {
            if (is_corner[i]) {
                corner_positions.push_back(i);
            }
        }
        
        // Compute minimum distance to any corner for each point
        for (int i = 0; i < npoints; ++i) {
            double min_dist = std::numeric_limits<double>::infinity();
            for (int corner_pos : corner_positions) {
                // Compute distance along polyline to this corner (both directions)
                double dist_forward = 0, dist_backward = 0;
                
                // Forward distance
                int curr = i;
                while (curr != corner_pos) {
                    int next = (curr + 1) % npoints;
                    dist_forward += segment_length(curr, next);
                    curr = next;
                }
                
                // Backward distance
                curr = i;
                while (curr != corner_pos) {
                    int prev = (curr - 1 + npoints) % npoints;
                    dist_backward += segment_length(prev, curr);
                    curr = prev;
                }
                
                min_dist = std::min(min_dist, std::min(dist_forward, dist_backward));
            }
            dist_to_closest_corner[i] = min_dist;
        }
    } else {
        // Original open polyline logic
        // Left-to-right pass
        double dist_from_last_corner = std::numeric_limits<double>::infinity();
        for (int i = 0; i < npoints; ++i) {
            if (is_corner[i]) {
                dist_from_last_corner = 0.0;
                dist_to_closest_corner[i] = 0.0;
            } else {
                if (dist_from_last_corner != std::numeric_limits<double>::infinity()) {
                    if (i > 0) {
                        dist_from_last_corner += segment_length(i-1, i);
                    }
                }
                dist_to_closest_corner[i] = dist_from_last_corner;
            }
        }

        // Right-to-left pass
        dist_from_last_corner = std::numeric_limits<double>::infinity();
        for (int i = npoints - 1; i >= 0; --i) {
            if (is_corner[i] > 0) {
                dist_from_last_corner = 0.0;
                dist_to_closest_corner[i] = 0.0;
            } else {
                if (dist_from_last_corner != std::numeric_limits<double>::infinity()) {
                    if (i < npoints - 1) {
                        dist_from_last_corner += segment_length(i, i+1);
                    }
                }
                // Take minimum of left-to-right and right-to-left distances
                dist_to_closest_corner[i] = std::min(dist_to_closest_corner[i], dist_from_last_corner);
            }
        }
    }
}

template<class F>
int find_closest_local_minimum(F&& f, int npoints, int p, int closed) {
    // Helper function to check if index i is a local minimum
    auto is_local_minimum = [&](int i) -> bool {
        int left_idx, right_idx;
        if (closed) {
            left_idx = (i - 1 + npoints) % npoints;
            right_idx = (i + 1) % npoints;
        } else {
            left_idx = i - 1;
            right_idx = i + 1;
        }
        
        bool left_ok = (!closed && i == 0) || (f(i) <= f(left_idx));
        bool right_ok = (!closed && i == npoints - 1) || (f(i) <= f(right_idx));
        return left_ok && right_ok;
    };

    // Check if starting point is already a local minimum
    if (is_local_minimum(p)) {
        return p;
    }

    // Walk outward from p to find closest local minimum
    int left = (p - 1 + npoints) % npoints;
    int right = (p + 1) % npoints;
    int distance = 1;

    while (distance < npoints) {
        // Check left candidate
        if (is_local_minimum(left)) {
            return left;
        }

        // Check right candidate  
        if (is_local_minimum(right)) {
            return right;
        }

        // Move outward
        distance++;
        if (closed) {
            left = (left - 1 + npoints) % npoints;
            right = (right + 1) % npoints;
            
            // Stop if we've covered the entire polyline
            if (distance >= npoints / 2) {
                break;
            }
        } else {
            left--;
            right++;
            
            // Stop if we're out of bounds
            if (left < 0 && right >= npoints) {
                break;
            }
        }
    }

    // This should never happen if the function is well-defined,
    // but if no local minimum is found, return the starting point
    return p;
}

int find_closest_to_focus(int npoints, const double* x, const double* y, double focus_x, double focus_y, int closed, int prev_closest_to_focus_idx = -1)
{
    auto focus_dist = [=](int i) {
        return euclidean_distance(x[i], y[i], focus_x, focus_y);
    };
    if(prev_closest_to_focus_idx < 0) {
        // no previous point - simply find the closest point to focus
        int closest_idx = 0;
        double min_dist = focus_dist(0);
        for (int i = 1; i < npoints; ++i) {
            double dist = focus_dist(i);
            if (dist < min_dist) {
                min_dist = dist;
                closest_idx = i;
            }
        }
        return closest_idx;
    }
    else if(!closed && prev_closest_to_focus_idx <= 0) {
        return 0;
    }
    else if(!closed && prev_closest_to_focus_idx >= npoints-1) {
        return npoints-1; //stick to endpoints once you reach them
                                          //(the user can always lift the stylus and click again to "get out of this"; OTOH
                                          //sliding away from endpoints makes it hard to pull them anywhere
    }
    else {
        //return the location of the local minimum of Euclidean distance to the focus point closest to prev_closest_to_focus_idx.
        //the idea here is to "stick to the area first selected" since otherwise if we look for the closest point every time,
        //and the polyline crosses itself, we will jump from part to part (or we might jump from one end of the polyline to the other)
        return find_closest_local_minimum(focus_dist, npoints, prev_closest_to_focus_idx, closed);
    }
}

void shortest_diff_with_wraparound(const std::vector<bool>& changed, int* first_diff, int* last_diff)
{
    int npoints = changed.size();
    // Check if we have a more efficient representation with wraparound
    // Find the longest consecutive sequence of unchanged points
    int max_unchanged_start = -1;
    int max_unchanged_length = 0;
    int current_unchanged_start = -1;
    int current_unchanged_length = 0;

    for (int i = 0; i < 2 * npoints; ++i) { // Go around twice to handle wraparound
        int idx = i % npoints;
        if (!changed[idx]) {
            if (current_unchanged_start == -1) {
                current_unchanged_start = idx;
                current_unchanged_length = 1;
            } else {
                current_unchanged_length++;
            }
        } else {
            if (current_unchanged_length > max_unchanged_length) {
                max_unchanged_length = current_unchanged_length;
                max_unchanged_start = current_unchanged_start;
            }
            current_unchanged_start = -1;
            current_unchanged_length = 0;
        }
    }

    // If the longest unchanged sequence is more than half the points,
    // we can represent the changes more efficiently
    if (max_unchanged_length > npoints / 2) {
        int unchanged_end = (max_unchanged_start + max_unchanged_length - 1) % npoints;
        *first_diff = (unchanged_end + 1) % npoints;
        *last_diff = (max_unchanged_start - 1 + npoints) % npoints;
    }
}

extern "C"
int old_smooth_polyline(int npoints, double* new_x, double* new_y, const double* x, const double* y,
                    double focus_x, double focus_y, int* first_diff, int* last_diff,
                    int prev_closest_to_focus_idx = -1,
                    const unsigned char* is_corner = nullptr, double corner_effect_strength = 0.5,
                    double threshold = 30.0, double smoothness = 0.6,
                    double pull_strength = 0.5, int num_neighbors = 1,
                    double max_endpoint_dist = 30.0, double zero_endpoint_dist_start = 5.0);
//TODO: optimize this to be "closer" to O(changed points) rather than O(N) - eg currently we compute stuff
//for points which we know will remain unchanged
extern "C"
int smooth_polyline(int closed, int npoints, double* new_x, double* new_y, const double* x, const double* y,
                    double focus_x, double focus_y, int* first_diff, int* last_diff,
                    int prev_closest_to_focus_idx = -1,
                    const unsigned char* is_corner = nullptr, double corner_effect_strength = 0.5,
                    double threshold = 30.0, double smoothness = 0.6,
                    double pull_strength = 0.5, int num_neighbors = 1,
                    double max_endpoint_dist = 30.0, double zero_endpoint_dist_start = 5.0) {
    *first_diff = -1;
    *last_diff = npoints;
    if (npoints < 2) {
        for (int i = 0; i < npoints; ++i) {
            new_x[i] = x[i];
            new_y[i] = y[i];
        }
        return 0;
    }

    auto segment_length = [&](int i1, int i2) {
        return euclidean_distance(x[i1], y[i1], x[i2], y[i2]);
    };

    int closest_idx = find_closest_to_focus(npoints, x, y, focus_x, focus_y, closed, prev_closest_to_focus_idx);

    // Compute polyline distances from closest point
    std::vector<double> polyline_dists;
    compute_polyline_distances(polyline_dists, npoints, segment_length, closest_idx, closed);

    // Compute distances from endpoints
    std::vector<double> dist_to_start(npoints);
    std::vector<double> dist_to_end(npoints);
    if(!closed) {
        dist_to_start[0] = 0;
        for (int i = 1; i < npoints; ++i) {
            dist_to_start[i] = dist_to_start[i-1] + segment_length(i-1, i);
        }
        dist_to_end[npoints-1] = 0;
        for (int i = npoints-2; i >= 0; --i) {
            dist_to_end[i] = dist_to_end[i+1] + segment_length(i, i+1);
        }
    }
    // Compute distances to closest corner points
    std::vector<double> dist_to_closest_corner;
    if (corner_effect_strength > 0) {
        compute_distances_to_closest_corner(dist_to_closest_corner, npoints, segment_length, closed, is_corner);
    } else {
        dist_to_closest_corner.resize(npoints, std::numeric_limits<double>::infinity());
    }

    auto effect_weight = [threshold](double poly_dist) {
        if (poly_dist >= threshold) return 0.0;
        double t = poly_dist / threshold;
        return 0.5 * (1.0 + std::cos(M_PI * t));
    };

    auto endpoint_effect_scale = [](double dist_to_endpoint, double endpoint_dist) {
        if (dist_to_endpoint >= endpoint_dist) return 1.0;
        double t = dist_to_endpoint / endpoint_dist;
        return t * t;
    };

    // Corner effect scale function - reduces smoothing near corners
    auto corner_effect_scale = [corner_effect_strength](double dist_to_corner, double corner_radius) {
        if (corner_effect_strength == 0.0 || dist_to_corner == std::numeric_limits<double>::infinity()) {
            return 1.0; // No corner effect
        }
        if (dist_to_corner >= corner_radius) {
            return 1.0; // Far from corner, full smoothing
        }
        double t = dist_to_corner / corner_radius;
        // Smooth transition from 0 (at corner) to 1 (at corner_radius distance)
        double base_scale = t * t * (3.0 - 2.0 * t); // Smooth step function
        return 1.0 - corner_effect_strength * (1.0 - base_scale);
    };

    // Compute pull direction
    double pull_dir_x = focus_x - x[closest_idx];
    double pull_dir_y = focus_y - y[closest_idx];
    double pull_magnitude = euclidean_distance(x[closest_idx], y[closest_idx], focus_x, focus_y);
    if (pull_magnitude > 0) {
        pull_dir_x /= pull_magnitude;
        pull_dir_y /= pull_magnitude;
    } else {
        pull_dir_x = pull_dir_y = 0.0;
    }

    // Endpoint distances are only relevant for open polylines
    double start_endpoint_dist = 0, end_endpoint_dist = 0;
    if (!closed) {
        start_endpoint_dist = std::min(max_endpoint_dist,
            euclidean_distance(focus_x, focus_y, x[0], y[0]) - zero_endpoint_dist_start);
        if(closest_idx == 0) {
            start_endpoint_dist = 0;
        }
        end_endpoint_dist = std::min(max_endpoint_dist,
            euclidean_distance(focus_x, focus_y, x[npoints-1], y[npoints-1]) - zero_endpoint_dist_start);
        if(closest_idx == npoints-1) {
            end_endpoint_dist = 0;
        }
    }

    // Use threshold as corner radius for determining corner effect range
    double corner_radius = threshold;

    // Track changes for efficient first_diff/last_diff in closed case
    std::vector<bool> changed(npoints, false);

    for (int i = 0; i < npoints; ++i) {
        double px = x[i], py = y[i];
        double poly_dist = polyline_dists[i];

        // Compute endpoint effect scale (only for open polylines)
        double endpoint_scale = 1.0;
        if(!closed) {
            if (dist_to_start[i] < start_endpoint_dist) {
                endpoint_scale = std::min(endpoint_scale,
                        endpoint_effect_scale(dist_to_start[i], start_endpoint_dist));
            }
            if (dist_to_end[i] < end_endpoint_dist) {
                endpoint_scale = std::min(endpoint_scale,
                        endpoint_effect_scale(dist_to_end[i], end_endpoint_dist));
            }
        }

        // Compute corner effect scale
        double corner_scale = corner_effect_scale(dist_to_closest_corner[i], corner_radius);

        // Smoothing component (now affected by corner proximity)
        double smooth_weight = std::min(effect_weight(poly_dist) * smoothness * endpoint_scale * corner_scale, 0.5);
        double smoothed_x = px, smoothed_y = py;
        if (smooth_weight > 0) {
            double sum_x = 0.0, sum_y = 0.0;
            int neighbor_count = 0;
            for (int j = 1; j <= num_neighbors; ++j) {
                int left_idx, right_idx;
                if (closed) {
                    left_idx = (i - j + npoints) % npoints;
                    right_idx = (i + j) % npoints;
                } else {
                    left_idx = i - j;
                    right_idx = i + j;
                }
                
                if (closed || left_idx >= 0) {
                    sum_x += x[left_idx];
                    sum_y += y[left_idx];
                    ++neighbor_count;
                }
                if (closed || right_idx < npoints) {
                    sum_x += x[right_idx];
                    sum_y += y[right_idx];
                    ++neighbor_count;
                }
            }
            if (neighbor_count > 0) {
                double avg_x = sum_x / neighbor_count;
                double avg_y = sum_y / neighbor_count;
                smoothed_x = px * (1.0 - smooth_weight) + avg_x * smooth_weight;
                smoothed_y = py * (1.0 - smooth_weight) + avg_y * smooth_weight;
            }
        }

        // Pull component (unchanged - not affected by corners)
        double pull = effect_weight(poly_dist) * pull_strength * endpoint_scale;
        if (pull > 0 && pull_magnitude > 0) {
            new_x[i] = smoothed_x + pull * pull_magnitude * pull_dir_x;
            new_y[i] = smoothed_y + pull * pull_magnitude * pull_dir_y;
        } else {
            new_x[i] = smoothed_x;
            new_y[i] = smoothed_y;
        }

        // Check for differences
        if (new_x[i] != x[i] || new_y[i] != y[i]) {
            changed[i] = true;
        }
    }
    
    // Compute first_diff and last_diff efficiently for both open and closed cases
    for (int i = 0; i < npoints; ++i) {
        if (changed[i]) {
            if (*first_diff == -1) {
                *first_diff = i;
            }
            *last_diff = i;
        }
    }
    
    // For closed polylines, handle wraparound case more efficiently
    if (closed && *first_diff != -1) {
        shortest_diff_with_wraparound(changed, first_diff, last_diff);
    }
    
    return closest_idx;
}
