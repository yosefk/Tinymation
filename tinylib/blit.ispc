
//a vectorized version of blit.cpp, with some additional optimizations
//(the outputs might differ, firstly because of the use of a floating point reciprocal,
//and secondly because of "255 - af" instead of 256 - af followed by min(out,255)

static inline int round8b(int val) { return (val+(1<<7))>>8; }
static inline int round16b(int val) { return (val+(1<<15))>>16; }

#define uint8_t uint8

export void blit_rgba8888(uniform uint32 bg_base[], const uniform uint32 fg_base[],
                          uniform int bg_stride, uniform int fg_stride, uniform int width, uniform int height,
                          uniform int bg_alpha, uniform int fg_alpha)
{
    if(bg_alpha == 255) {
        bg_alpha = 256;
    }
    if(fg_alpha == 255) {
        fg_alpha = 256;
    }

    for(uniform int y=0; y<height; ++y) {
        uniform uint32* bg = bg_base + (bg_stride*y>>2);
        const uint32* uniform fg = fg_base + (fg_stride*y>>2);

        foreach(x=0 ... width) {
            uint32 bgpix = bg[x];
            uint32 fgpix = fg[x];
            int ab = ((bgpix>>24) * bg_alpha) >> 8;
            int af = ((fgpix>>24) * fg_alpha) >> 8;
            int af8 = af << 8;
            int ab1minusaf = ab*(255 - af);
            int ao = round8b(af8 + ab1minusaf);
            
            uint32 opix = ao << 24;

            ao = ao == 0 ? 1 : ao;
            float finv = 1.0f / ao;

            int rb = bgpix & 0xff;
            int rf = fgpix & 0xff;
            int gb = (bgpix >> 8) & 0xff;
            int gf = (fgpix >> 8) & 0xff;
            int bb = (bgpix >> 16) & 0xff;
            int bf = (fgpix >> 16) & 0xff;
                
            int ro = round8b((int)((float)(af8*rf + ab1minusaf*rb) * finv));
            int go = round8b((int)((float)(af8*gf + ab1minusaf*gb) * finv));
            int bo = round8b((int)((float)(af8*bf + ab1minusaf*bb) * finv));

            bg[x] = opix | ro | (go<<8) | (bo<<16);
        }
    }
}
