import numpy as np
from PySide6.QtGui import QImage
import ctypes as ct
from tinylib_ctypes import tinylib, SurfaceToBlit, RangeFunc

import time

COLOR_UNINIT = 'uninit'

def strides_preserving_copy(a):
    '''np.copy "normalizes" strides instead of copying them so we have this function to avoid it'''
    like = np.empty_like(a)
    like[...] = a[...]
    return like

class Stat:
    def __init__(self, op, what):
        self.op = op
        self.what = what
        self.ns = 0
        self.things = 0
        self.calls = 0
    def start(self):
        self.start_ns = time.perf_counter_ns()
    def stop(self, things):
        self.ns += time.perf_counter_ns() - self.start_ns
        self.things += things
        self.calls += 1
    def show(self):
        if self.things:
            print(f'{self.op}: {self.ns / self.things} ns/{self.what} ({self.calls} calls)')

stats = []
def stat(op, what='pixel'):
    s = Stat(op, what)
    stats.append(s)
    return s

def show_stats():
    for s in stats:
        s.show()

blit_stat = stat('Surface.blit')
blits_stat = stat('Surface.blits')
fill_stat = stat('Surface.fill')
copy_stat = stat('Surface.copy')
blend_stat = stat('Surface.blend')
load_stat = stat('surf.load')
png_save_stat = stat('surf.save(png)')
uncompressed_save_stat = stat('surf.save(uncompressed)')
get_mask_stat = stat('get_mask')
combine_mask_alphas_stat = stat('combine_mask_alphas')
held_mask_stat = stat('held_mask')
scale_inter_area_stat = stat('scale(INTER_AREA)')
scale_inter_linear_stat = stat('scale(INTER_LINEAR)')
scale_inter_cubic_stat = stat('scale(INTER_CUBIC)')
rotate_stat = stat('rotate')

def surf_array(w, h):
    return np.ndarray((w, h, 4), strides=(4, w*4, 1), dtype=np.uint8)

# if the code were written from scratch, rather than adapted from a pygame.Surface-based implementation,
# it might have made sense to stick with the numpy "height, width, channels" convention, different
# from the typical (and pygame's) "width, height, channels" convention as it is, because our "non-standard"
# (for numpy) strides hurt numpy ops throughput.
class Surface:
    def __init__(self, size_or_data, alpha=255, base=None, color=None):
        self._alpha = alpha
        self._base = base
        if type(size_or_data) is tuple:
            w, h = size_or_data
            w, h = round(w), round(h)
            self._a = surf_array(w, h)
            if color!=COLOR_UNINIT:
                self.fill(color if color is not None else (0,0,0,0))
        else:
            self._a = size_or_data
            w, h, channels = size_or_data.shape
            assert color is None
            assert channels == 4
            assert size_or_data.strides[0] == 4
            assert size_or_data.strides[1] >= w * 4
            assert size_or_data.strides[2] == 1

    def get_width(self):
        return self._a.shape[0]

    def get_height(self):
        return self._a.shape[1]

    def get_rect(self):
        return (0, 0, self.get_width(), self.get_height())

    def get_size(self):
        return self._a.shape[:2]

    def bytes_per_line(self):
        return self._a.strides[1]

    def fill(self, color):
        if len(color) == 3:
            color = tuple(color) + (255,)

        #this is slow maybe since we have a "non-C-contiguous array"?.. and definitely because it's 4 elements and not 1
        #in testing this is very slow even with contiguous arrays
        #self._a[:,:] = np.array(color) 

        rgba = color[0] | (color[1]<<8) | (color[2]<<16) | (color[3]<<24)
        w, h = self.get_size()

        fill_stat.start()
        # this is a bit faster than ispc and a lot faster than assigning np.array(color); it doesn't seem to matter if we transpose the dimensions or not
        self.uint32_unsafe()[:,:] = rgba
        fill_stat.stop(w*h)

    def blend(self, color):
        assert len(color) == 4
        blend_stat.start()
        w,h = self.get_size()
        @RangeFunc
        def blend_tile(start_y, finish_y):
            tinylib.blend_rgb_copy_alpha(self.base_ptr(), self.bytes_per_line(), w, start_y, finish_y, *color)
        tinylib.parallel_for_grain(blend_tile, 0, h, 0 if (w*h > 500000) else h)
        blend_stat.stop(w*h)

    def base_ptr(self):
        if self._base is None:
            self._base = self._a.ctypes.data_as(ct.c_void_p)
        return self._base.value

    def _ptr_to(self, x, y):
        return ct.c_void_p(self.base_ptr() + y * self.bytes_per_line() + x * 4)

    def _blit_args(background, foreground, xy):
        x, y = xy
        x, y = round(x), round(y)
        xw = min(x + foreground.get_width(), background.get_width())
        yh = min(y + foreground.get_height(), background.get_height())

        fg_x_oft = max(0, -x)
        fg_y_oft = max(0, -y)
        x = max(x, 0)
        y = max(y, 0)

        blitw = xw - x
        blith = yh - y
        assert x + blitw <= background.get_width()
        assert y + blith <= background.get_height()
        assert fg_x_oft + blitw <= foreground.get_width()
        assert fg_y_oft + blith <= foreground.get_height()

        return x, y, fg_x_oft, fg_y_oft, blitw, blith

    def blit(background, foreground, xy=(0,0), rect=None, into=None):
        assert rect is None or rect == foreground.get_rect() # whatever rect does in pygame, tinymation never really used it

        bgx, bgy, fgx, fgy, blitw, blith = background._blit_args(foreground, xy)

        blit_stat.start()

        @RangeFunc
        def blit_tile(start_y, finish_y):
            if into is None: 
                tinylib.blit_rgba8888_inplace(background._ptr_to(bgx, bgy), foreground._ptr_to(fgx, fgy),
                                              background.bytes_per_line(), foreground.bytes_per_line(),
                                              blitw, start_y, finish_y,
                                              background.get_alpha(), foreground.get_alpha())
            else:
                tinylib.blit_rgba8888(background._ptr_to(bgx, bgy), foreground._ptr_to(fgx, fgy), into._ptr_to(bgx, bgy),
                                      background.bytes_per_line(), foreground.bytes_per_line(), into.bytes_per_line(),
                                      blitw, start_y, finish_y,
                                      background.get_alpha(), foreground.get_alpha())

        tinylib.parallel_for_grain(blit_tile, 0, blith, 0 if (blith*blitw > 500000) else blith)

        blit_stat.stop(blitw*blith)

    def blits(background, foregrounds, xy=(0,0)):
        '''logically equivalent to, but faster than (2x-ish thanks to less cache spills):

        for fg in foregrounds:
            background.blit(fg, xy)
        '''

        if not foregrounds:
            return
        sizes = [fg.get_size() for fg in foregrounds]
        for size in sizes:
            assert size == sizes[0], sizes

        bgx, bgy, fgx, fgy, blitw, blith = background._blit_args(foregrounds[0], xy)
        surfaces_to_blit = (SurfaceToBlit * (len(foregrounds) + 1))()
        
        bg = surfaces_to_blit[0]
        bg.base = background._ptr_to(bgx, bgy)
        bg.stride = background.bytes_per_line()
        bg.alpha = background._alpha

        i = 1
        for foreground in foregrounds:
            fg = surfaces_to_blit[i]
            fg.base = foreground._ptr_to(fgx, fgy)
            fg.stride = foreground.bytes_per_line()
            fg.alpha = foreground._alpha
            i += 1

        blits_stat.start()

        @RangeFunc
        def blit_tile(start_y, finish_y):
            tinylib.blits_rgba8888_inplace(surfaces_to_blit, len(surfaces_to_blit), blitw, start_y, finish_y)
        tinylib.parallel_for_grain(blit_tile, 0, blith, 0 if (blitw*blith*len(surfaces_to_blit) > 500000) else blith) 

        blits_stat.stop(blitw * blith * len(foregrounds))

    def is_contig_transposed(self):
        return self.get_width()*4 == self.bytes_per_line()

    def copy(self):
        copy_stat.start()
        s = Surface(strides_preserving_copy(self._a), alpha=self._alpha)
        # interestingly this is slower, not faster:
        #s = Surface(self.get_size(), alpha=self._alpha, color=COLOR_UNINIT)
        #s.trans_unsafe()[...] = self.trans_unsafe()[...]
        copy_stat.stop(self.get_width()*self.get_height())
        return s

    def empty_like(self):
        return Surface((self.get_width(), self.get_height()), alpha=self._alpha, color=COLOR_UNINIT)

    def subsurface(self, *args, clip=False):
        assert len(args) in [1, 4]
        if len(args) == 1:
            x,y,w,h = [round(i) for i in args[0]]
        else:
            x,y,w,h = [round(i) for i in args]

        if not clip:
            if x < 0 or y < 0 or x+w > self.get_width() or y+h > self.get_height():
                raise ValueError(f"subsurface rectangle ({x},{y},{w},{h}) outside surface area ({self.get_size()})")

        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        w = max(w, 0)
        h = max(h, 0)
        return Surface(self._a[x:x+w,y:y+h,:], alpha=self._alpha, base=self._ptr_to(x,y))

    def set_alpha(self, alpha):
        assert alpha >= 0 and alpha < 256
        self._alpha = int(alpha)

    def get_alpha(self):
        return self._alpha

    def get_at(self, pos):
        x,y = pos
        return tuple([int(c) for c in self._a[x,y]])

    def qimage_unsafe(self):
        '''this QImage is aliased to the array - don't use after freeing the array/surface object'''
        w, h, _ = self._a.shape
        bytes_per_line = self.bytes_per_line()
        assert w*4 == bytes_per_line # note that we don't support non-contig surfaces - but copy().qimage_[unsafe]() will work
        # without the cast, if we just pass ptr_to(0,0), we get garbage pixel data, I wonder what's happening there
        ibuffer = ct.cast(self.base_ptr(), ct.POINTER(ct.c_uint8 * (h * bytes_per_line * 4))).contents
        return QImage(ibuffer, w, h, bytes_per_line, QImage.Format_RGBA8888)

    def qimage(self): return self.qimage_unsafe().copy()

    def trans_unsafe(self):
        w, h, _ = self._a.shape
        bytes_per_line = self.bytes_per_line()
        ibuffer = ct.cast(self.base_ptr(), ct.POINTER(ct.c_uint8 * (h * bytes_per_line * 4))).contents
        return np.ndarray((h,w,4), buffer=ibuffer, strides=(bytes_per_line,4,1), dtype=np.uint8)

    def trans_uint32_unsafe(self):
        w, h, _ = self._a.shape
        bytes_per_line = self.bytes_per_line()
        ibuffer = ct.cast(self.base_ptr(), ct.POINTER(ct.c_uint32 * (h * bytes_per_line))).contents
        return np.ndarray((h,w), buffer=ibuffer, strides=(bytes_per_line,4), dtype=np.uint32)

    def uint32_unsafe(self):
        w, h, _ = self._a.shape
        bytes_per_line = self.bytes_per_line()
        ibuffer = ct.cast(self.base_ptr(), ct.POINTER(ct.c_uint32 * (h * bytes_per_line))).contents
        return np.ndarray((w,h), buffer=ibuffer, strides=(4,bytes_per_line), dtype=np.uint32)

def load(fname):

    load_stat.start()

    img = QImage(fname)
    
    if img.isNull():
        raise ValueError(f"Failed to load image from {fname}")
    
    # Convert to RGBA format
    # TODO: maybe we could do better and force loading into the right format?..

    # Qt Docs on Format_RGBA8888:
    # The image is stored using a 32-bit byte-ordered RGBA format (8-8-8-8). Unlike ARGB32 this is a byte-ordered format,
    # which means the 32bit encoding differs between big endian and little endian architectures, being respectively (0xRRGGBBAA) and (0xAABBGGRR).
    # The order of the colors is the same on any architecture if read as bytes 0xRR,0xGG,0xBB,0xAA.
    if img.format() != QImage.Format_RGBA8888:
        img = img.convertToFormat(QImage.Format_RGBA8888)
    
    width = img.width()
    height = img.height()
    
    ptr = img.bits()
    arr = np.ndarray((width, height, 4), buffer=ptr, strides=(4, width*4, 1), dtype=np.uint8)

    ret = Surface(strides_preserving_copy(arr))

    load_stat.stop(width*height)

    return ret

def save(surface, filename):
    ext = filename.split('.')[-1]
    stat = png_save_stat if ext=='png' else uncompressed_save_stat
    stat.start()
    surface.qimage_unsafe().save(filename)
    stat.stop(surface.get_width() * surface.get_height())

def rotate(surface, angle):
    assert angle in [90, -90]
    rotate_stat.start()
    rotated = np.rot90(surface._a, {90:1,-90:-1}[angle])
    width,height = surface.get_size()
    arr = surf_array(height, width)
    arr[...] = rotated[...]
    rotate_stat.stop(width*height)
    return Surface(arr, surface.get_alpha())

def pixels3d(surface):
    return surface._a[:,:,0:3]

def pixels_alpha(surface):
    return surface._a[:,:,3]

def box(surface, rect, color):
    sub = surface.subsurface(rect, clip=True)
    if len(color) == 3 or color[3] == 255:
        sub.fill(color[:3])
    else:
        sub.blend(color)

def rect(surface, color, rect, width=0, border_radius=0):
    if width == 0 and border_radius == 0:
        box(surface, rect, color)
    else:
        if len(color)==3:
            color = color + (255,)
        x,y,w,h = [round(n) for n in rect]
        sw,sh = surface.get_size()
        if w <= 0 or h <= 0:
            return

        rgba = (ct.c_uint8*4)(*color)
        tinylib.fill_rounded_rectangle_negative(surface.base_ptr(), surface.bytes_per_line(), sw, sh, x, y, w, h,
                                                float(width), border_radius, rgba)

def filled_circle(surface, cx, cy, radius, color):
    '''a very inefficient filled_circle, needn't do better atm since this is only done for cursors at init time'''
    # Create output RGBA array, initialized to transparent (0, 0, 0, 0)
    tmp_surf = Surface(surface.get_size())
    img = tmp_surf._a
    width = surface.get_width()
    height = surface.get_height()
    
    # Create meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
    
    # Calculate distance from each pixel to the center
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Anti-aliasing: smooth transition over a 1-pixel boundary
    # Fully opaque inside (distance <= radius - 0.5)
    # Linearly interpolated alpha between radius - 0.5 and radius + 0.5
    # Transparent outside (distance > radius + 0.5)
    alpha = np.clip(0.5 - (distance - radius), 0, 1)
    
    # Set color where alpha > 0
    mask = alpha > 0
    for i in range(3):  # Set RGB channels
        img[mask, i] = color[i]
    img[mask, 3] = alpha[mask] * color[3]  # Alpha channel, scaled by input alpha

    surface.blit(tmp_surf)
    

