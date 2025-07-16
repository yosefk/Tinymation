import numpy as np
from PySide6.QtGui import QImage
import numpy.ctypeslib as npct
import ctypes as ct
tinylib = npct.load_library('tinylib','.')

import time

#extern "C" void blit_rgba8888(uint8_t* __restrict bg_base, const uint8_t* __restrict fg_base,
#                              int bg_stride, int fg_stride, int width, int height,
#                              int bg_alpha, int fg_alpha)
tinylib.blit_rgba8888_inplace.argtypes = [ct.c_void_p]*2 + [ct.c_int]*6
tinylib.blit_rgba8888.argtypes = [ct.c_void_p]*3 + [ct.c_int]*7
tinylib.fill_32b.argtypes = [ct.c_void_p] + [ct.c_int]*3 + [ct.c_uint]

SRCALPHA = 'srcalpha'
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
        self.start_ns = time.time_ns()
    def stop(self, things):
        self.ns += time.time_ns() - self.start_ns
        self.things += things
        self.calls += 1
    def show(self):
        if self.things:
            print(f'{self.op}: {self.ns / self.things} ns/{self.what}')

stats = []
def stat(op, what):
    s = Stat(op, what)
    stats.append(s)
    return s

def show_stats():
    for s in stats:
        s.show()

blit_stat = stat('Surface.blit','pixel')
fill_stat = stat('Surface.fill','pixel')

# if the code were written from scratch, rather than adapted from a pygame.Surface-based implementation,
# it might have made sense to stick with the numpy "height, width, channels" convention, different
# from the typical (and pygame's) "width, height, channels" convention as it is, because our "non-standard"
# (for numpy) strides hurt numpy ops throughput.
class Surface:
    def __init__(self, size_or_data, srcalpha=None, alpha=255, base=None, color=None):
        self._alpha = alpha
        self._base = base
        if type(size_or_data) is tuple:
            w, h = size_or_data
            w, h = round(w), round(h)
            self._a = np.ndarray((w, h, 4), strides=(4, w*4, 1), dtype=np.uint8)
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
        assert srcalpha in [None, SRCALPHA]

    def get_width(self):
        return self._a.shape[0]

    def get_height(self):
        return self._a.shape[1]

    def get_rect(self):
        return (0, 0, self.get_width(), self.get_height())

    def bytes_per_line(self):
        return self._a.strides[1]

    def fill(self, color):
        if len(color) == 3:
            color = tuple(color) + (255,)
        fill_stat.start()

        #this is slow since we have a "non-C-contiguous array"?.. or just because it's 4 elements and not 1?..
        #self._a[:,:] = np.array(color) 

        rgba = color[0] | (color[1]<<8) | (color[2]<<16) | (color[3]<<24)
        tinylib.fill_32b(self._ptr_to(0,0), self.get_width(), self.get_height(), self.bytes_per_line(), rgba)

        fill_stat.stop(self.get_width()*self.get_height())

    def _ptr_to(self, x, y):
        if self._base is None:
            self._base = self._a.ctypes.data_as(ct.c_void_p)
        return ct.c_void_p(self._base.value + y * self.bytes_per_line() + x * 4)

    def blit(background, foreground, xy=(0,0), rect=None, into=None):
        assert rect is None or rect == foreground.get_rect() # whatever rect does in pygame, tinymation never really used it
        x, y = xy
        x, y = round(x), round(y)
        xw = min(x + foreground.get_width(), background.get_width())
        yh = min(y + foreground.get_height(), background.get_height())

        fg_x_oft = max(0, -x)
        fg_y_oft = max(0, -y)
        x = max(x, 0)
        y = max(y, 0)

        # FIXME: add assertions on sizes!!

        blit_stat.start()

        if into is None: 
            tinylib.blit_rgba8888_inplace(background._ptr_to(x,y), foreground._ptr_to(fg_x_oft, fg_y_oft),
                                          background.bytes_per_line(), foreground.bytes_per_line(),
                                          xw - x, yh - y,
                                          background.get_alpha(), foreground.get_alpha())
        else:
            tinylib.blit_rgba8888(background._ptr_to(x,y), foreground._ptr_to(fg_x_oft, fg_y_oft), into._ptr_to(x,y),
                                  background.bytes_per_line(), foreground.bytes_per_line(), into.bytes_per_line(),
                                  xw - x, yh - y,
                                  background.get_alpha(), foreground.get_alpha())

        blit_stat.stop((xw-x)*(yh-y))

    def blits(self, blit_sequence):
        for args in blit_sequence:
            self.blit(*args)

    def copy(self):
        return Surface(strides_preserving_copy(self._a), alpha=self._alpha)

    def empty_like(self):
        return Surface((self.get_width(), self.get_height()), alpha=self._alpha, color=COLOR_UNINIT)

    def subsurface(self, *args):
        assert len(args) in [1, 4]
        if len(args) == 1:
            x,y,w,h = [round(i) for i in args[0]]
        else:
            x,y,w,h = [round(i) for i in args]
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
        # without the cast, if we just pass ptr_to(0,0), we get garbage pixel data, I wonder what's happening there
        ibuffer = ct.cast(self._ptr_to(0,0), ct.POINTER(ct.c_uint8 * (w * bytes_per_line * 4))).contents
        return QImage(ibuffer, w, h, bytes_per_line, QImage.Format_RGBA8888)

    def qimage(self): return self.qimage_unsafe().copy()

def load(fname):
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

    return Surface(strides_preserving_copy(arr))

def save(surface, filename):
    surface.qimage_unsafe().save(filename)

#    return pg.image.save(surface.surface, filename)

def rotate(surface, angle):
    return surface # FIXME
#    return Surface(pg.transform.rotate(surface.surface, angle))

def pixels3d(surface):
    return surface._a[:,:,0:3]

def pixels_alpha(surface):
    return surface._a[:,:,3]

def box(surface, rect, color):
    assert len(color)==3
    surface.subsurface(rect).fill(color)

def rect(surface, color, rect, *rest):
    pass
#    return pg.draw.rect(surface.surface, color, rect, *rest)

def filled_circle(surface, x, y, radius, color):
    pass
#    return pg.gfxdraw.filled_circle(surface.surface, x, y, radius, color)

