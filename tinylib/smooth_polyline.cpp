#define _USE_MATH_DEFINES
#include <vector>
#include <cmath>
#include <algorithm>

extern "C"
void smooth_polyline(int npoints, double* new_x, double* new_y, const double* x, const double* y,
                    double focus_x, double focus_y, int* first_diff, int* last_diff,
                    const double* curvature_maximums = nullptr, double corner_effect_strength = 0.5,
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
        return;
    }

    auto euclidean_distance = [](double x1, double y1, double x2, double y2) {
        return std::sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
    };

    auto segment_length = [&](int i1, int i2) {
        return euclidean_distance(x[i1], y[i1], x[i2], y[i2]);
    };

    // Find closest point to focus
    int closest_idx = 0;
    double min_dist = euclidean_distance(x[0], y[0], focus_x, focus_y);
    for (int i = 1; i < npoints; ++i) {
        double dist = euclidean_distance(x[i], y[i], focus_x, focus_y);
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
        }
    }

    // Compute polyline distances from closest point
    std::vector<double> polyline_dists(npoints);
    polyline_dists[closest_idx] = 0;
    for (int i = closest_idx + 1; i < npoints; ++i) {
        polyline_dists[i] = polyline_dists[i-1] + segment_length(i-1, i);
    }
    for (int i = closest_idx - 1; i >= 0; --i) {
        polyline_dists[i] = polyline_dists[i+1] + segment_length(i, i+1);
    }

    // Compute distances from endpoints
    std::vector<double> dist_to_start(npoints);
    std::vector<double> dist_to_end(npoints);
    dist_to_start[0] = 0;
    for (int i = 1; i < npoints; ++i) {
        dist_to_start[i] = dist_to_start[i-1] + segment_length(i-1, i);
    }
    dist_to_end[npoints-1] = 0;
    for (int i = npoints-2; i >= 0; --i) {
        dist_to_end[i] = dist_to_end[i+1] + segment_length(i, i+1);
    }

    // Compute distances to closest corner points
    std::vector<double> dist_to_closest_corner(npoints, std::numeric_limits<double>::infinity());
    
    if (curvature_maximums != nullptr && corner_effect_strength > 0) {
        // Left-to-right pass
        double dist_from_last_corner = std::numeric_limits<double>::infinity();
        for (int i = 0; i < npoints; ++i) {
            if (curvature_maximums[i] > 0) {
                // This is a corner point
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
            if (curvature_maximums[i] > 0) {
                // This is a corner point
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

    double start_endpoint_dist = std::min(max_endpoint_dist,
        euclidean_distance(focus_x, focus_y, x[0], y[0]) - zero_endpoint_dist_start);
    double end_endpoint_dist = std::min(max_endpoint_dist,
        euclidean_distance(focus_x, focus_y, x[npoints-1], y[npoints-1]) - zero_endpoint_dist_start);

    // Use threshold as corner radius for determining corner effect range
    double corner_radius = threshold;

    for (int i = 0; i < npoints; ++i) {
        double px = x[i], py = y[i];
        double poly_dist = polyline_dists[i];

        // Compute endpoint effect scale
        double endpoint_scale = 1.0;
        if (dist_to_start[i] < start_endpoint_dist) {
            endpoint_scale = std::min(endpoint_scale,
                endpoint_effect_scale(dist_to_start[i], start_endpoint_dist));
        }
        if (dist_to_end[i] < end_endpoint_dist) {
            endpoint_scale = std::min(endpoint_scale,
                endpoint_effect_scale(dist_to_end[i], end_endpoint_dist));
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
                if (i - j >= 0) {
                    sum_x += x[i-j];
                    sum_y += y[i-j];
                    ++neighbor_count;
                }
            }
            for (int j = 1; j <= num_neighbors; ++j) {
                if (i + j < npoints) {
                    sum_x += x[i+j];
                    sum_y += y[i+j];
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
            if (*first_diff == -1) {
                *first_diff = i;
            }
            *last_diff = i;
        }
    }
}
