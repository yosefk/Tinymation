
//"standard" alpha blending for 8b images
//
//we blit the foreground image onto the background image (in-place) according to the formula:
//
//  ao = af+ab*(1-af)
//  co = (af*cf + (1-af)*ab*cb)/ao
//
//...were af,cf, ab,cb, and ao,co are the alpha and color values of the foreground, background and output image, respectively.
//
//we scale the alpha values of the foreground and the background images by global parameter values (to support
//a global "layer transparency" setting in addition to the per-pixel alpha value.)
//
//properties of this code:
//
//* looks the same as Krita normal mode blending (no bit-level comparison done, just copied a bunch of layers
//  with transparent colors into Krita and looked) unlike pygame blitting (with or without BLEND_ALPHA_SDL2)
//* blitting is associative or "roughly associative" (again based on eyeballing how layers look like with different
//  layer selected as the current, which changes the blitting order in Tinymation since the bottom and top layers
//  are "pre-blitted" and cached and then you blit bottom-cache, current, top-cache in that order), again unlike
//  pygame blitting with or without BLEND_ALPHA_SDL2
//* about 10x slower than pygame blitting (without BLEND_ALPHA_SDL2 which is what Tinymation used to be using),
//  in part because it refuses to auto-vectorize as described below
//* supports global alpha values in addition to the alpha masks (unlike Qt/QPainter, which additionally doens't
//  promise high performance except for premultiplied alpha)

static inline int round8b(int val) { return (val+(1<<7))>>8; }
static inline int round16b(int val) { return (val+(1<<15))>>16; }

#define uint8_t uint8

export void blit_rgba8888(uniform uint32 bg_base[], const uniform uint32 fg_base[],
                          uniform int bg_stride, uniform int fg_stride, uniform int width, uniform int height,
                          uniform int bg_alpha, uniform int fg_alpha)
{
    //we use >>8 instead of /255 and since 255 is a common "global layer transparency" value,
    //we prefer for it to have exactly no effect instead of almost no effect on the output:
    if(bg_alpha == 255) {
        bg_alpha = 256;
    }
    if(fg_alpha == 255) {
        fg_alpha = 256;
    }

    for(uniform int y=0; y<height; ++y) {
        uniform uint32* bg = bg_base + (bg_stride*y>>2);
        const uint32* uniform fg = fg_base + (fg_stride*y>>2);

        //this loop doesn't auto-vectorize under g++ for 2 reasons:
        //* it is worried about read/write data dependencies despite __restrict
        //  (which can be solved by writing to a temporary row buffer)
        //* it doesn't like the division, and an inverse LUT is met with
        //  "not suitable for gather load" (?!).

        foreach(x=0 ... width) {
            /*
            uint32 bgpix = bg[x];
            uint32 fgpix = fg[x];
            uint8 ab = round8b((bgpix >> 24) * bg_alpha);
            uint8 af = round8b((fgpix >> 24) * fg_alpha);
            uint16 ao16 = round8b((af<<8) + ab*(256 - af));

            uint32 opix = (ao16>255 ? 255 : ao16) << 24;

            uint16 ao = ao16 == 0 ? 1 : ao16;
            float finv = 1.0f / ao;
            for(int c=0; c<3*8; c+=8) {
                uint8 cb = (bgpix >> c) & 0xff;
                uint8 cf = (fgpix >> c) & 0xff;
                int co = round8b((int)((float)((af*cf<<8) + (256-af)*ab*cb) * finv));

                uint8 co8 = co>255 ? 255 : co;

                opix |= co8 << c;
            }

            bg[x] = opix;
            */

            uint32 bgpix = bg[x];
            uint32 fgpix = fg[x];
            int ab = round8b((bgpix>>24) * bg_alpha);
            int af = round8b((fgpix>>24) * fg_alpha);
            int ao = round8b((af<<8) + ab*(256 - af));
            
            uint32 opix = (ao>255 ? 255 : ao)<<24;

            ao = ao == 0 ? 1 : ao;
            float finv = 1.0f / ao;
//            int inv = finv * 65536.f;
            for(int c=0; c<3; ++c) {
                int cb = (bgpix >> c*8) & 0xff;
                int cf = (fgpix >> c*8) & 0xff;
                int co = round8b((int)((float)((af*cf<<8) + (256-af)*ab*cb) * finv));

                co = co>255 ? 255 : co;

                opix |= co << c*8;
            }

            bg[x] = opix;
        }
    }
}
