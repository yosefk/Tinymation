/*
 Please write a C++ function for filling "the negative of a rounded rectangle". Meaning, instead of filling a rectangle with rounded corners (thus leaving the corners unfilled relatively to filling a "normal" rectangle), fill the bounding rectangle of the given rounded rectangle (specified by width, height, border width - if it's zero, then only pixels near the rounded corners get filled, and if it's positive, the unfilled part "shrinks" by the given number of pixels from both sides), and a radius defining the rounded corners size), leaving the rounded rectangle itself unfilled. All that with antialiasing.The function prototype should be: void fill_rounded_rectangle_negative(unsigned uint8_t* rgba, int image_stride, int image_width, int image_height, int x, int y, int rect_width, int rect_height, float border_width, float corner_radius, const uint8_t* color); the value of the RGB channel ch (one of R,G or B) of a pixel x,y is given by rgba[image_stridey+x4+ch]. you can assume the rectangle (which starts at x, y and ends at x+rect_width,y+rect_height (exclusive, not inclusive for the latter coordinate) fits into the image (from (0,0) to (image_width,image_height)). you can ignore the alpha channel of the input image. the input color has the format RGBA like the pixels of the input image. if its alpha value is 255, you blend it with the RGB values (mostly you take the original pixels - inside the rectangle - or the color value - outside it; right at the boundary you blend for anti-aliasing.) if the input color has an alpha value below 255, then you use this alpha value to blend with the input image pixels wherever you touch them. (use the same formula for when the color alpha is 255 or smaller; you don't need 2 branches in the code, I just used "if this, if that" to explain how this should work)

Great! Could you optimize this to not iterate over most of the pixels? There are basically 8, usually small sub-rectangles where we have any chance to have to set a pixel: the 4 corners (squares of size radius x radius; we assume that border_width is smaller than the radius) and then the 4 sides "between the corners", where border_width+ up to 1 pixel for antialiasing might be affected. Could you iterate over these, and tailor the distance computation to these areas? Reuse code to not make it very long but don't compromise efficiency too much like using one distance field everywhere
 */

#include <cmath>
#include <algorithm>
#include <cstdio>

extern "C"

void fill_rounded_rectangle_negative(unsigned char* rgba, int image_stride, int image_width, int image_height,
                                   int x, int y, int rect_width, int rect_height,
                                   float border_width, float corner_radius, const unsigned char* color) {

    // Extract color components
    float color_r = color[0] / 255.0f;
    float color_g = color[1] / 255.0f;
    float color_b = color[2] / 255.0f;
    float color_a = color[3] / 255.0f;

    // Calculate inner rectangle (shrunk by border_width)
    float inner_x = x + border_width;
    float inner_y = y + border_width;
    float inner_w = rect_width - 2.0f * border_width;
    float inner_h = rect_height - 2.0f * border_width;
    float inner_radius = std::max(0.0f, corner_radius - border_width);

    // Helper to blend a pixel
    auto blend_pixel = [&](int px, int py, float coverage) {
	if(px < 0 || py < 0 || px >= image_width || py >= image_height) {
	    return;
        }
        int pixel_idx = image_stride * py + px * 4;
        float current_r = rgba[pixel_idx + 0] / 255.0f;
        float current_g = rgba[pixel_idx + 1] / 255.0f;
        float current_b = rgba[pixel_idx + 2] / 255.0f;

        float blend_factor = coverage * color_a;
        float final_r = current_r * (1.0f - blend_factor) + color_r * blend_factor;
        float final_g = current_g * (1.0f - blend_factor) + color_g * blend_factor;
        float final_b = current_b * (1.0f - blend_factor) + color_b * blend_factor;

        rgba[pixel_idx + 0] = (unsigned char)(final_r * 255.0f + 0.5f);
        rgba[pixel_idx + 1] = (unsigned char)(final_g * 255.0f + 0.5f);
        rgba[pixel_idx + 2] = (unsigned char)(final_b * 255.0f + 0.5f);
    };

    // If inner rectangle is invalid, fill everything
    if (inner_w <= 0.0f || inner_h <= 0.0f) {
        for (int py = y; py < y + rect_height; py++) {
            for (int px = x; px < x + rect_width; px++) {
		blend_pixel(px, py, 1);
            }
        }
        return;
    }

    // Calculate corner positions and sizes
    int corner_size = (int)std::ceil(corner_radius + 1.0f);
    int border_size = (int)std::ceil(border_width + 1.0f);

    // Corner centers in inner rectangle coordinates
    float corner_left = inner_x + inner_radius;
    float corner_right = inner_x + inner_w - inner_radius;
    float corner_top = inner_y + inner_radius;
    float corner_bottom = inner_y + inner_h - inner_radius;

    // Process the 4 corners
    auto process_corner = [&](int start_x, int start_y, int end_x, int end_y, float cx, float cy) {
        for (int py = start_y; py < end_y; py++) {
            for (int px = start_x; px < end_x; px++) {
                float dx = px + 0.5f - cx;
                float dy = py + 0.5f - cy;
                float dist = std::sqrt(dx * dx + dy * dy) - inner_radius;
                float coverage = std::clamp(dist + 0.5f, 0.0f, 1.0f);
                blend_pixel(px, py, coverage);
            }
        }
    };

    // Top-left corner
    process_corner(x, y,
                  std::min(x + corner_size, (int)std::ceil(corner_left)),
                  std::min(y + corner_size, (int)std::ceil(corner_top)),
                  corner_left, corner_top);

    // Top-right corner
    process_corner(std::max(x + rect_width - corner_size, (int)std::floor(corner_right)), y,
                  x + rect_width,
                  std::min(y + corner_size, (int)std::ceil(corner_top)),
                  corner_right, corner_top);

    // Bottom-left corner
    process_corner(x, std::max(y + rect_height - corner_size, (int)std::floor(corner_bottom)),
                  std::min(x + corner_size, (int)std::ceil(corner_left)),
                  y + rect_height,
                  corner_left, corner_bottom);

    // Bottom-right corner
    process_corner(std::max(x + rect_width - corner_size, (int)std::floor(corner_right)),
                  std::max(y + rect_height - corner_size, (int)std::floor(corner_bottom)),
                  x + rect_width, y + rect_height,
                  corner_right, corner_bottom);

    // Process the 4 sides (edges between corners)

    // Top edge
    int top_start_x = std::min(x + corner_size, (int)std::ceil(corner_left));
    int top_end_x = std::max(x + rect_width - corner_size, (int)std::floor(corner_right));
    if (top_start_x < top_end_x) {
        for (int py = y; py < std::min(y + border_size, (int)std::ceil(inner_y)); py++) {
            for (int px = top_start_x; px < top_end_x; px++) {
                float dist = inner_y - (py + 0.5f);
                float coverage = std::clamp(dist + 0.5f, 0.0f, 1.0f);
                blend_pixel(px, py, coverage);
            }
        }
    }

    // Bottom edge
    int bottom_start_x = std::min(x + corner_size, (int)std::ceil(corner_left));
    int bottom_end_x = std::max(x + rect_width - corner_size, (int)std::floor(corner_right));
    if (bottom_start_x < bottom_end_x) {
        for (int py = std::max(y + rect_height - border_size, (int)std::floor(inner_y + inner_h));
             py < y + rect_height; py++) {
            for (int px = bottom_start_x; px < bottom_end_x; px++) {
                float dist = (py + 0.5f) - (inner_y + inner_h);
                float coverage = std::clamp(dist + 0.5f, 0.0f, 1.0f);
                blend_pixel(px, py, coverage);
            }
        }
    }

    // Left edge
    int left_start_y = std::min(y + corner_size, (int)std::ceil(corner_top));
    int left_end_y = std::max(y + rect_height - corner_size, (int)std::floor(corner_bottom));
    if (left_start_y < left_end_y) {
        for (int py = left_start_y; py < left_end_y; py++) {
            for (int px = x; px < std::min(x + border_size, (int)std::ceil(inner_x)); px++) {
                float dist = inner_x - (px + 0.5f);
                float coverage = std::clamp(dist + 0.5f, 0.0f, 1.0f);
                blend_pixel(px, py, coverage);
            }
        }
    }

    // Right edge
    int right_start_y = std::min(y + corner_size, (int)std::ceil(corner_top));
    int right_end_y = std::max(y + rect_height - corner_size, (int)std::floor(corner_bottom));
    if (right_start_y < right_end_y) {
        for (int py = right_start_y; py < right_end_y; py++) {
            for (int px = std::max(x + rect_width - border_size, (int)std::floor(inner_x + inner_w));
                 px < x + rect_width; px++) {
                float dist = (px + 0.5f) - (inner_x + inner_w);
                float coverage = std::clamp(dist + 0.5f, 0.0f, 1.0f);
                blend_pixel(px, py, coverage);
            }
        }
    }
}
