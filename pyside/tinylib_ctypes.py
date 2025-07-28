import ctypes as ct
import tbb # for tinylib.parallel_for_grain
import numpy.ctypeslib as npct
tinylib = npct.load_library('tinylib','.')

import os
WORKERS = min(os.cpu_count(), 8)


#Brush* brush_init_paint(double x, double y, double time, double pressure, double lineWidth, double smoothDist, int dry, int erase, int softLines,
#                        unsigned char* image, int width, int height, int xstride, int ystride, const int* paintWithinRegion)
tinylib.brush_init_paint.argtypes = [ct.c_double]*6 + [ct.c_int]*3 + [ct.c_void_p] + [ct.c_int]*4 + [ct.c_void_p]
tinylib.brush_init_paint.restype = ct.c_void_p

#void brush_set_rgb(Brush* brush, const unsigned char* rgb)
tinylib.brush_set_rgb.argtypes = [ct.c_void_p, ct.c_uint8*3]

#void brush_paint(Brush* brush, int npoints, double* x, double* y, const double* time, const double* pressure, double zoom, int* region)
tinylib.brush_paint.argtypes = [ct.c_void_p, ct.c_int] + [ct.c_void_p]*4 + [ct.c_double] + [ct.c_void_p]

#void brush_end_paint(Brush* brush, int* region)
tinylib.brush_end_paint.argtypes = [ct.c_void_p]*2

#int brush_get_polyline_length(Brush* brush)
tinylib.brush_get_polyline_length.argtypes = [ct.c_void_p]
tinylib.brush_get_polyline_length.restype = ct.c_int

#void brush_get_polyline(Brush* brush, int polyline_length, double* polyline_x, double* polyline_y, double* polyline_time, double* polyline_pressure)
tinylib.brush_get_polyline.argtypes = [ct.c_void_p, ct.c_int] + [ct.c_void_p]*4

#void brush_free(Brush* brush)
tinylib.brush_free.argtypes = [ct.c_void_p]

#void brush_flood_fill_color_based_on_mask(Brush* brush, int* color, unsigned char* mask, int color_stride, int mask_stride,
#                                          int _8_connectivity, int mask_new_val, int new_color_value)
tinylib.brush_flood_fill_color_based_on_mask.argtypes = [ct.c_void_p]*3 + [ct.c_int]*5

tinylib.fitpack_parcur.argtypes = [ct.c_void_p]*2 + [ct.c_int]*3 + [ct.c_double] + [ct.c_void_p]*3

#extern "C" void blit_rgba8888(uint8_t* __restrict bg_base, const uint8_t* __restrict fg_base,
#                              int bg_stride, int fg_stride, int width, int start_y, int finish_y,
#                              int bg_alpha, int fg_alpha)
tinylib.blits_rgba8888_inplace.argtypes = [ct.c_void_p] + [ct.c_int]*3
tinylib.blit_rgba8888_inplace.argtypes = [ct.c_void_p]*2 + [ct.c_int]*7
tinylib.blit_rgba8888.argtypes = [ct.c_void_p]*3 + [ct.c_int]*8
#export void blend_rgb_copy_alpha(uniform uint32 base[],  uniform int stride, uniform int width, uniform int start_y, uniform int finish_y,
#                                 uniform int r, uniform int g, uniform int b, uniform int a)
tinylib.blend_rgb_copy_alpha.argtypes = [ct.c_void_p] + [ct.c_int]*8

class SurfaceToBlit(ct.Structure):
    _fields_ = [
        ('base', ct.c_void_p),
        ('stride', ct.c_int),
        ('alpha', ct.c_int),
    ]

class LayerParamsForMask(ct.Structure):
    _fields_ = [
        ('lines_base', ct.c_void_p),
        ('color_base', ct.c_void_p),
        ('lines_stride', ct.c_int),
        ('color_stride', ct.c_int),
        ('lines_lit', ct.c_int),
    ]
#export void blit_layers_mask(uniform LayerParamsForMask layers[], uniform int n,
#                             uniform uint8 mask_base[], uniform int mask_stride, uniform int width, uniform int start_y, uniform int finish_y)
tinylib.blit_layers_mask.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p] + [ct.c_int]*4

class MaskAlphaParams(ct.Structure):
    _fields_ = [
        ('base', ct.c_void_p),
        ('stride', ct.c_int),
        ('rgb', ct.c_uint32),
    ]

#export void blit_combined_mask(uniform MaskAlphaParams mask_alphas[], uniform int n,
#                               uniform uint32 mask_base[], uniform int mask_stride, uniform int width, uniform int start_y, uniform int finish_y)
tinylib.blit_combined_mask.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p] + [ct.c_int]*4

#export void blit_held_mask(uniform uint32 mask_base[], uniform int mask_stride,
#              const uniform uint32 curr_layer_base[], uniform int curr_layer_stride,
#	           const uniform uint32 held_inside_base[], uniform int held_inside_stride,
#			   const uniform uint32 held_outside_base[], uniform int held_outside_stride,
#			   const uniform uint32 rest_base[], uniform int rest_stride,
#			   uniform uint32 tmp_row[], //can't call new[], getting a "uniform" type error
#			   uniform int width, uniform int start_y, uniform int finish_y)
tinylib.blit_held_mask.argtypes = [ct.c_void_p, ct.c_int]*5 + [ct.c_void_p] + [ct.c_int]*3

tinylib.fill_32b.argtypes = [ct.c_void_p] + [ct.c_int]*4 + [ct.c_uint]

#void fill_rounded_rectangle_negative(uint8_t* rgba, int image_stride, int image_width, int image_height,
#                                    int x, int y, int rect_width, int rect_height, float border_width,
#                                    float corner_radius, const uint8_t* color) {
tinylib.fill_rounded_rectangle_negative.argtypes = [ct.c_void_p] + [ct.c_int]*7 + [ct.c_float]*2 + [ct.c_uint8*4]

RangeFunc = ct.CFUNCTYPE(None, ct.c_int, ct.c_int)
tinylib.parallel_for_grain.argtypes = [RangeFunc] + [ct.c_int]*3

tinylib.parallel_set_num_threads.argtypes = [ct.c_int]

tinylib.parallel_set_num_threads(WORKERS)

