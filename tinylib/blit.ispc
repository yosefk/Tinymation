
//a vectorized version of blit.cpp, with some additional optimizations
//(the outputs might differ, firstly because of the use of a floating point reciprocal,
//and secondly because of "255 - af" instead of 256 - af followed by min(out,255)

static inline int round8b(int val) { return (val+(1<<7))>>8; }
static inline int round16b(int val) { return (val+(1<<15))>>16; }

//we do this because ISPC aliasing rules say that pointers are always "restrict"/unaliased
//and it thus sounds scary to have a helper inline function getting 2 pointers that might be
//the same pointers, as bg_base and out_base are in the in-place blitting case.
#define BLIT_BODY(out_base,out_stride) \
    if(bg_alpha == 255) { \
        bg_alpha = 256; \
    } \
    if(fg_alpha == 255) { \
        fg_alpha = 256; \
    } \
 \
    for(uniform int y=0; y<height; ++y) { \
        uint32* uniform out = out_base + (out_stride*y>>2); \
        uniform uint32* bg = bg_base + (bg_stride*y>>2); \
        const uint32* uniform fg = fg_base + (fg_stride*y>>2); \
 \
        foreach(x=0 ... width) { \
            uint32 bgpix = bg[x]; \
            uint32 fgpix = fg[x]; \
            int ab = ((bgpix>>24) * bg_alpha) >> 8; \
            int af = ((fgpix>>24) * fg_alpha) >> 8; \
            int af8 = af << 8; \
            int ab1minusaf = ab*(255 - af); \
            int ao = round8b(af8 + ab1minusaf); \
 \
            uint32 opix = ao << 24; \
\
            ao = ao == 0 ? 1 : ao; \
            float finv = 1.0f / ao; \
 \
            int rb = bgpix & 0xff; \
            int rf = fgpix & 0xff; \
            int gb = (bgpix >> 8) & 0xff; \
            int gf = (fgpix >> 8) & 0xff; \
            int bb = (bgpix >> 16) & 0xff; \
            int bf = (fgpix >> 16) & 0xff; \
 \
            int ro = round8b((int)((float)(af8*rf + ab1minusaf*rb) * finv)); \
            int go = round8b((int)((float)(af8*gf + ab1minusaf*gb) * finv)); \
            int bo = round8b((int)((float)(af8*bf + ab1minusaf*bb) * finv)); \
 \
            out[x] = opix | ro | (go<<8) | (bo<<16); \
        } \
    }

export void blit_rgba8888_inplace(uniform uint32 bg_base[], const uniform uint32 fg_base[],
                          uniform int bg_stride, uniform int fg_stride, uniform int width, uniform int height,
                          uniform int bg_alpha, uniform int fg_alpha)
{
    BLIT_BODY(bg_base, bg_stride)
}

export void blit_rgba8888(uniform uint32 bg_base[], const uniform uint32 fg_base[], uniform uint32 out_base[],
                          uniform int bg_stride, uniform int fg_stride, uniform int out_stride,
			  uniform int width, uniform int height,
                          uniform int bg_alpha, uniform int fg_alpha)
{
    BLIT_BODY(out_base, out_stride)
}

struct SurfaceToBlit
{
    uniform uint32* base;
    uniform int stride;
    uniform int alpha;
};

export void blits_rgba8888_inplace(uniform SurfaceToBlit surfaces[], uniform int n, uniform int width, uniform int height)
{
    //assuming a 16K L1 d-cache, we can fit 2 rows of 1920 pixels into the cache;
    //one from the background and one from one of the foregrounds.
    //so we iterate over all the ys, and blit all the foreground rows onto the background row,
    //in the hope to avoid evicting the background and then having to read it again for every foreground.
    for(uniform int y=0; y<height; ++y) {
	uniform SurfaceToBlit& bg = surfaces[0];
	uniform uint32* uniform bg_base = bg.base + (bg.stride * y >> 2);
    	for(uniform int i=1; i<n; ++i) {
	    uniform SurfaceToBlit& fg = surfaces[i];
	    //the for SurfaceToBlit& as opposed to SurfaceToBlit* to satisfy "uniform/
	    //varying" type reqs is peculiar
            uniform uint32* uniform fg_base = fg.base + (fg.stride * y >> 2);
	    blit_rgba8888_inplace(bg_base, fg_base, 0, 0, width, 1, bg.alpha, fg.alpha);
	}
    }
}

//blend between the RGB values of the input array with the given RGB value (giving it the weight a
//and the input RGB values the weight 255-a), and copy the alpha as is from the input to the output
export void blend_rgb_copy_alpha(uniform uint32 base[],  uniform int stride, uniform int width, uniform int height,
                                 uniform int r, uniform int g, uniform int b, uniform int a)
{
    for(uniform int y=0; y<height; ++y) {
        uniform uint32* row = base + (stride*y>>2);
 
        foreach(x=0 ... width) {
            uint32 pix = row[x];
            int ai = pix >> 24;

            int ri = pix & 0xff;
            int gi = (pix >> 8) & 0xff;
            int bi = (pix >> 16) & 0xff;

            int ro = round8b(r * a + ri * (255-a));
            int go = round8b(g * a + gi * (255-a));
            int bo = round8b(b * a + bi * (255-a));

            row[x] = ro | (go<<8) | (bo<<16) | (ai<<24);
        }
    }
}

void fill_row(uniform uint32 base[], uniform int len, uniform uint32 value)
{
    foreach(x=0 ... len) {
    	base[x] = value;
    }
}

export void fill_32b(uniform uint32 base[], uniform int width, uniform int height, uniform int bytesPerLine,
                     uniform uint32 value)
{
    uniform int stride32 = bytesPerLine >> 2;
    if(stride32 == width) {
    	fill_row(base, width*height, value);
    }
    else {
    	for(uniform int y=0; y<height; ++y) {
	    fill_row(base+stride32*y, width, value);
	}
    }
}
