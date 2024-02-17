// pygame.Surface RGBA images are stored s.t. a single RGBA pixel is contiguous in memory,
// and the pixels of a row follow each other in memory. We hardcode these assumptions here
// (and test them at runtime when converting numpy arrays returned by pygame.surfarray.{pixels3d(), pixels_alpha()}
// functions to C function arguments. Moreover, stride appears to be always equal to width * 4
// but we don't assume it (and presumably it's not true when getting a subsurface, for example,
// while the rows and the pixels stay contiguous.)
//
// an annoying detail is that pygame.Surface can be both RGBA and BGRA; in particular, pygame.image.load()
// returns RGBA surfaces while pygame.Surface returns BGRA surfaces.
//
// note that pixels3d() as well as pixels_alpha() return arrays with a stride of 4 between
// the pixels, and treating a pixels3d() array as a 4D RGBA array "does the job" (=you appear
// to be able to access the alpha channel), although the pygame Surface API does not provide
// a 4D array. [It could be that the reason for this is that with a 4D array, there's no way
// to make the difference between RGBA and BGRA transparent; with a separate alpha array,
// you can make BGR look like RGB using a stride of -1. It doesn't work with an alpha channel
// (it could only make ABGR look like RGBA, but not BGRA.)]

#include <stdio.h>

extern "C" void meshgrid_color(unsigned char* color, int stride, int width, int height, int bgr)
{
	//printf("color=%p stride=%d width=%d height=%d width*4=%d bgr=%d\n", color, stride, width, height, width*4, bgr);
	int r = bgr ? 2 : 0;
	int g = 1;
	int b = bgr ? 0 : 2;
	int a = 3;
	for(int y=0; y<height; ++y) {
		unsigned char* row = color + y*stride;
		for(int x4=0; x4<width*4; x4+=4) {
			row[x4+r] = y*255/height;
			row[x4+g] = x4*255/(width*4);
			row[x4+b] = 0;
			//here we treat the RGB array as an RGBA array...
			if((x4 >> 7) % 2) {
				row[x4+a] = y*255/height;
			}
			else {
				row[x4+a] = x4*255/(width*4);
			}
		}
	}
}

extern "C" void meshgrid_alpha(unsigned char* alpha, int stride, int width, int height)
{
	for(int y=0; y<height; ++y) {
		unsigned char* row = alpha + y*stride;
		for(int x4=0; x4<width*4; x4+=4) {
			if((y >> 5) % 2) {
				row[x4] = y*255/height;
			}
			else {
				row[x4] = x4*255/(width*4);
			}
		}
	}
}
