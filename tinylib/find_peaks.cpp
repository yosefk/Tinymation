
extern "C" void find_peaks(unsigned char* peaks, const double* y, int n, double height, int distance) {
    // Check each point to see if it's a peak
    for (int i = 0; i < n; i++) {
        // First check if height requirement is met
        if (y[i] <= height) {
            peaks[i] = 0;
            continue;
        }
        
        // Check if this point is higher than all points within distance
        bool is_peak = true;
        int start = (i - distance < 0) ? 0 : i - distance;
        int end = (i + distance >= n) ? n - 1 : i + distance;
        
        for (int j = start; j <= end; j++) {
            // Skip comparing with itself
            if (j == i) {
                continue;
            }
            
            // If any point within distance is >= current point, it's not a peak
            if (y[j] >= y[i]) {
                is_peak = false;
                break;
            }
        }
        
        peaks[i] = is_peak ? 1 : 0;
    }
}
