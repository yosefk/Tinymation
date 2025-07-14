#include <cstdint>
#include <algorithm>

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

extern "C" void blit_rgba8888(uint8_t* __restrict bg_base, const uint8_t* __restrict fg_base,
                              int bg_stride, int fg_stride, int width, int height,
                              int bg_alpha, int fg_alpha)
{
    //we use >>8 instead of /255 and since 255 is a common "global layer transparency" value,
    //we prefer for it to have exactly no effect instead of almost no effect on the output:
    if(bg_alpha == 255) {
        bg_alpha = 256;
    }
    if(fg_alpha == 255) {
        fg_alpha = 256;
    }

    for(int y=0; y<height; ++y) {
        uint8_t* __restrict bg = bg_base + bg_stride*y;
        const uint8_t* __restrict fg = fg_base + fg_stride*y;

        //this loop doesn't auto-vectorize under g++ for 2 reasons:
        //* it is worried about read/write data dependencies despite __restrict
        //  (which can be solved by writing to a temporary row buffer)
        //* it doesn't like the division, and an inverse LUT is met with
        //  "not suitable for gather load" (?!).

        for(int x=0; x<width*4; x+=4) {
            int ab = round8b(bg[x+3] * bg_alpha);
            int af = round8b(fg[x+3] * fg_alpha);
            int ao = round8b((af<<8) + ab*(256 - af));
            
            bg[x+3] = std::min(255, ao);

            ao = ao == 0 ? 1 : ao;
            for(int c=0; c<3; ++c) {
                int cb = bg[x+c];
                int cf = fg[x+c];
                int co = std::min(255, round8b(((af*cf<<8) + (256-af)*ab*cb) / ao));

                bg[x+c] = co;
            }
        }
    }
}
