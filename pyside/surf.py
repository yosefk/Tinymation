import numpy as np
from PySide6.QtGui import QImage
import numpy.ctypeslib as npct
import ctypes as ct
tinylib = npct.load_library('tinylib','.')

#extern "C" void blit_rgba8888(uint8_t* __restrict bg_base, const uint8_t* __restrict fg_base,
#                              int bg_stride, int fg_stride, int width, int height,
#                              int bg_alpha, int fg_alpha)
tinylib.blit_rgba8888.argtypes = [ct.c_void_p]*2 + [ct.c_int]*6

SRCALPHA = 'srcalpha'

def strides_preserving_copy(a):
    '''np.copy "normalizes" strides instead of copying them so we have this function to avoid it'''
    like = np.empty_like(a)
    like[...] = a[...]
    return like


totpix = 0
totns = 0

import time

class Surface:
    def __init__(self, size_or_data, srcalpha=None, alpha=255, clip=None):
        if type(size_or_data) is tuple:
            w, h = size_or_data
            w, h = round(w), round(h)
            self._a = np.ndarray((w, h, 4), strides=(4, w*4, 1), dtype=np.uint8)
            self._a[:] = 0
        else:
            self._a = size_or_data
            w, h, channels = size_or_data.shape
            assert channels == 4
            assert size_or_data.strides[0] == 4
            assert size_or_data.strides[1] >= w * 4
            assert size_or_data.strides[2] == 1
        assert srcalpha in [None, SRCALPHA]
        # currently all of our surfaces are RGBA, 8 bit per channel.
        # the couple of cases where we use RGB pygame surfaces should work as RGBA, too.
        # eventually we'll probably want 16b RGBA surfaces for blitting multiple transparent layers
        # without a non-transparent background
        self._clip = (0, 0, w, h) if clip is None else clip
        self._alpha = alpha

    def get_width(self):
        return self._a.shape[0]

    def get_height(self):
        return self._a.shape[1]

    def get_rect(self):
        return (0, 0, self.get_width(), self.get_height())

    def fill(self, color):
        if len(color) == 3:
            color = tuple(color) + (255,)
        x, y, w, h = self._clip
        self._a[x:x+w,y:y+h] = np.array(color)

    def blit(background, foreground, xy=(0,0), rect=None):
        assert rect is None or rect == foreground.get_rect() # whatever rect does in pygame, tinymation never really used it
        x, y = xy
        x, y = round(x), round(y)
        xw = min(x + foreground.get_width(), background.get_width())
        yh = min(y + foreground.get_height(), background.get_height())
        x = max(x, 0)
        y = max(y, 0)

        start = time.time_ns()

        bg_base = background._a.ctypes.data_as(ct.c_void_p)
        bg_base = ct.c_void_p(bg_base.value + y * background._a.strides[1] + x * background._a.strides[0])
        tinylib.blit_rgba8888(bg_base, foreground._a.ctypes.data_as(ct.c_void_p),
                              background._a.strides[1], foreground._a.strides[1],
                              xw - x, yh - y,
                              background.get_alpha(), foreground.get_alpha())

        global totns
        global totpix
        totns += time.time_ns() - start
        totpix += (xw-x)*(yh-y)

    def blits(self, blit_sequence):
        for args in blit_sequence:
            self.blit(*args)

    def copy(self):
        return Surface(strides_preserving_copy(self._a), alpha=self._alpha, clip=self._clip)

    def subsurface(self, *args):
        assert len(args) in [1, 4]
        if len(args) == 1:
            x,y,w,h = [round(i) for i in args[0]]
        else:
            x,y,w,h = [round(i) for i in args]
        assert self._clip == self.get_rect()
        return Surface(self._a[x:x+w,y:y+h,:], alpha=self._alpha)

    def set_alpha(self, alpha):
        assert alpha >= 0 and alpha < 256
        self._alpha = int(alpha)

    def get_alpha(self):
        return self._alpha

    def get_at(self, pos):
        x,y = pos
        return tuple([int(c) for c in self._a[x,y]])

    def set_clip(self, rect):
        self._clip = rect

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
    # FIXME
    pass
#    return pg.image.save(surface.surface, filename)

def rotate(surface, angle):
    return surface # FIXME
#    return Surface(pg.transform.rotate(surface.surface, angle))

def pixels3d(surface):
    return surface._a[:,:,0:3]

def pixels_alpha(surface):
    return surface._a[:,:,3]

# FIXME 
def box(surface, rect, color):
    pass
#    return pg.gfxdraw.box(surface.surface, rect, color)

def rect(surface, color, rect, *rest):
    pass
#    return pg.draw.rect(surface.surface, color, rect, *rest)

def filled_circle(surface, x, y, radius, color):
    pass
#    return pg.gfxdraw.filled_circle(surface.surface, x, y, radius, color)

def stat():
    print(f'totpix {totpix} totns {totns} ns/pix {totns/totpix}')
