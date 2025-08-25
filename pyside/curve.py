import math
import numpy as np
import ctypes as ct
from tinylib_ctypes import tinylib
import msgpack

class BrushConfig:
    def __init__(self, minPressureWidth, maxPressureWidth=None):
        self.minPressureWidth = minPressureWidth
        self.maxPressureWidth = maxPressureWidth if maxPressureWidth is not None else minPressureWidth

    def to_dict(self): return dict(minw=self.minPressureWidth, maxw=self.maxPressureWidth)
    @staticmethod
    def from_dict(d):
        b = BrushConfig(0)
        b.minPressureWidth = d['minw']
        b.maxPressureWidth = d['maxw']
        return b

    def copy(self): return BrushConfig.from_dict(self.to_dict())

def polyline_array(n, dtype=np.float32):
    return np.empty((n, 3), dtype=dtype, order='F')

def polyline_bbox(polyline):
    '''returns integer coordinates xmin, ymin [inclusive], xmax, ymax [exclusive]
    containing all the xy coordinates of the polyline'''
    xy = polyline[:,:2]
    xmin, ymin = np.min(xy, axis=0)
    xmax, ymax = np.max(xy, axis=0)
    eps = 0.01
    return math.floor(xmin), math.floor(ymin), math.ceil(xmax+eps), math.ceil(ymax+eps)

def compress_polyline(polyline, bbox=None):
    '''represent x, y, and pressure using 16 bit each. for pressure it's an overkill ATM since you
    get 16K pressure values at most and usually way less; for x,y, on a large graphics tablet you
    get 100K+ values which is a bit more than 64K but not by much. With an HD resolution,
    We get >30 coordinate values per pixel which has got to be pretty good'''
    if bbox is not None:
        bbox = polyline_bbox(polyline)
    xmin, ymin, xmax, ymax = bbox

    arr = polyline_array(len(polyline), dtype=np.uint16)
    maxval = 2**16-1
    arr[:,0] = np.round(maxval*(polyline[:,0] - xmin)/(xmax - xmin))
    arr[:,1] = np.round(maxval*(polyline[:,1] - ymin)/(ymax - ymin))
    arr[:,2] = maxval*polyline[:,2] # pressure is between 0 and 1

    return arr.tobytes(order='F'), bbox

def uncompress_polyline(polyline_bytes, bbox):
    polyline = np.frombuffer(polyline_bytes, dtype=np.uint16)
    polyline = polyline.reshape((len(polyline)//3, 3), order='F').astype(np.float32)

    arr = polyline_array(len(polyline))
    imaxval = 1/(2**16-1)

    xmin, ymin, xmax, ymax = bbox

    arr[:,0] = imaxval*polyline[:,0]*(xmax - xmin) + xmin
    arr[:,1] = imaxval*polyline[:,1]*(ymax - ymin) + ymin
    arr[:,2] = imaxval*polyline[:,2]

    return arr

def arr_base_ptr(arr): return arr.ctypes.data_as(ct.c_void_p)
def ptr_plus_oft(ptr, oft): return ct.c_void_p(ptr.value + oft)

class Curve:
    def __init__(self, polyline, closed, brushConfig):
        self._polyline32 = polyline.astype(np.float32) if polyline is not None else None
        self.closed = closed
        self.brushConfig = brushConfig

    def byte_size(self): return self._polyline32.nbytes
    def calc_bbox(self): return polyline_bbox(self._polyline32)
    def pixels_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        mw = self.brushConfig.maxPressureWidth + 3
        return xmin-mw, ymin-mw, xmax+mw, ymax+mw

    def to_dict(self, bbox=None):
        arr, bbox = compress_polyline(self._polyline32, bbox)
        return dict(closed=self.closed, brush=self.brushConfig.to_dict(), bbox=bbox, polyline=arr)

    @staticmethod
    def from_dict(d):
        c = Curve(None,None,None)
        c.closed = d['closed']
        c.brushConfig = BrushConfig.from_dict(d['brush'])
        bbox = d['bbox']
        c._polyline32 = uncompress_polyline(d['polyline'], bbox)
        return c, bbox

    def copy(self):
        return Curve(self._polyline32.copy(), self.closed, self.brushConfig.copy())

    def polyline32(self): return self._polyline32
    def polyline64(self): return self._polyline32.astype(float)

    def num_line_segments(self): return len(self._polyline32) - 1 + int(self.closed)
    def pressure2LineWidth(self, pressure):
        return self.brushConfig.minPressureWidth * (1-pressure) + self.brushConfig.maxPressureWidth * pressure;
    def line_segments_into(self, xstarts, ystarts, wstarts, xends, yends, wends):
        p = self._polyline32
        e = len(p)-1
        xstarts[:e] = p[:-1,0]
        xends[:e] = p[1:,0]
        ystarts[:e] = p[:-1,1]
        yends[:e] = p[1:,1]
        wstarts[:e] = self.pressure2LineWidth(p[:-1,2])
        wends[:e] = self.pressure2LineWidth(p[1:,2])
        if self.closed:
            xstarts[-1] = p[-1,0]
            xends[-1] = p[0,0]
            ystarts[-1] = p[-1,1]
            yends[-1] = p[0,1]
            wstarts[-1] = self.pressure2LineWidth(p[-1,2])
            wends[-1] = self.pressure2LineWidth(p[0,2])

    def render(self, alpha, paintWithin=None, get_polyline=False):
        n = len(self._polyline32)
        if n == 0:
            return

        arr = self.polyline64()
        x, y, p = arr[0]
        ptr = alpha.base_ptr()
        width, height = alpha.get_size()
        ystride = alpha.bytes_per_line()
        brush = tinylib.brush_init_paint(x, y, 0, p, self.brushConfig.minPressureWidth, self.brushConfig.maxPressureWidth, 0, 0, 0, 0, # smoothDist, dry, eraser and soft are all 0
                                         ptr, width, height, 1, ystride, arr_base_ptr(paintWithin) if paintWithin is not None else 0)

        if self.closed:
            tinylib.brush_set_closed(brush)

        xarr = arr[1:, 0]
        yarr = arr[1:, 1]
        parr = arr[1:, 2]

        tinylib.brush_paint(brush, n-1, arr_base_ptr(xarr), arr_base_ptr(yarr), 0, arr_base_ptr(parr), 1, 0)
        tinylib.brush_end_paint(brush, 0)

        retval = None
        if get_polyline:
            polyline_length = tinylib.brush_get_polyline_length(brush)
            polyline = polyline_array(polyline_length, dtype=float)
            polyline_x = polyline[:, 0]
            polyline_y = polyline[:, 1]
            polyline_p = polyline[:, 2]
            tinylib.brush_get_polyline(brush, polyline_length, arr_base_ptr(polyline_x), arr_base_ptr(polyline_y), 0, arr_base_ptr(polyline_p))

            nsamples = len(self._polyline32)+int(self.closed)
            sample2polyline = np.empty(nsamples, dtype=np.int32)
            tinylib.brush_get_sample2polyline(brush, nsamples, arr_base_ptr(sample2polyline))

            retval = (polyline, sample2polyline)

        tinylib.brush_free(brush)
        return retval

def bbox_array(n):
    return np.empty((n,4), order='F', dtype=np.int16)

def bbox_base_ptrs(arr):
    assert arr.dtype == np.int16
    n, dim = arr.shape
    assert dim == 4
    stride = arr.strides[1]
    p = arr_base_ptr(arr)
    return [ptr_plus_oft(p, i*stride) for i in range(4)]

def rectangles_intersect(rect1, rect2):
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2

    # Check if one rectangle is to the left of the other
    if x1_max < x2_min or x2_max < x1_min:
        return False

    # Check if one rectangle is above the other
    if y1_max < y2_min or y2_max < y1_min:
        return False

    return True

class CurveSet:
    def __init__(self):
        self.curves = []
        # these are polyline points bboxes - expand each by its curve's maxPressureWidth brush setting plus a margin to get a bbox
        # of the points affected by the rendering of the polyline
        self.bboxes = bbox_array(0)
    def append_curve(self, curve):
        self.curves.append(curve)
        old_bboxes = self.bboxes
        self.bboxes = bbox_array(len(self.curves))
        self.bboxes[0:len(old_bboxes)] = old_bboxes
        self.bboxes[len(old_bboxes)] = curve.calc_bbox()
    def pop_curve(self):
        self.bboxes = self.bboxes[:-1]
        return self.curves.pop()
    def replace_curve(self, index, curve):
        prev_curve = self.curves[index]
        self.curves[index] = curve
        self.bboxes[index] = curve.calc_bbox()
        return prev_curve
    def size(self): return len(self.curves)

    def render(self, alpha, paintWithin=None):
        for i, curve in enumerate(self.curves):
            if paintWithin is not None and not rectangles_intersect(paintWithin, curve.pixels_bbox(self.bboxes[i])):
                continue # this curve is not affecting the paintWithin region
            curve.render(alpha, paintWithin)

    def closest_curve_index(self, x, y):
        # 1. find the bboxes which could contain x,y
        n = self.size()
        indexes = np.empty(n, dtype=np.int32)
        nrelevant = tinylib.cull_bboxes(round(x), round(y), *bbox_base_ptrs(self.bboxes), n, arr_base_ptr(indexes))

        indexes = indexes[:nrelevant]

        # 2. find distances to the line segments of all polylines in the relevant bboxes
        ns = sum([self.curves[i].num_line_segments() for i in indexes])

        xstarts = np.empty(ns, dtype=np.float32)
        xends = np.empty(ns, dtype=np.float32)
        ystarts = np.empty(ns, dtype=np.float32)
        yends = np.empty(ns, dtype=np.float32)
        wstarts = np.empty(ns, dtype=np.float32)
        wends = np.empty(ns, dtype=np.float32)

        s = 0
        for i in indexes:
            curve = self.curves[i]
            e = s + curve.num_line_segments()
            curve.line_segments_into(xstarts[s:e], ystarts[s:e], wstarts[s:e], xends[s:e], yends[s:e], wends[s:e])
            s = e

        distances = np.empty(ns, dtype=np.float32)
        tinylib.point_to_curve_segments_distances(ns, arr_base_ptr(distances),
                                                  arr_base_ptr(xstarts), arr_base_ptr(ystarts), arr_base_ptr(wstarts),
                                                  arr_base_ptr(xends), arr_base_ptr(yends), arr_base_ptr(wends),
                                                  x, y)

        closest_segment_ind = np.argmin(distances)
        sum_num_segments = 0
        for i in indexes:
            sum_num_segments += self.curves[i].num_line_segments()
            if closest_segment_ind < sum_num_segments:
                return i

        assert False, "{closest_segment_ind=} does not belong to any polyline, each having the segment numbers {[self.curves[i].num_line_segments() for i in indexes]}"

    def to_list(self):
        return [curve.to_dict(bbox) for (curve, bbox) in zip(self.curves,self.bboxes)]
    @staticmethod
    def from_list(ls):
        c = CurveSet()
        curves_and_bboxes = [Curve.from_dict(d) for d in ls]
        if not curves_and_bboxes:
            return c
        c.curves, bboxes = zip(*curves_and_bboxes)
        c.curves = list(c.curves)
        c.bboxes = bbox_array(len(ls))
        for i, bbox in enumerate(bboxes):
            c.bboxes[i] = bbox
        return c

    def copy(self):
        c = CurveSet()
        c.curves = [curve.copy() for curve in self.curves]
        c.bboxes = self.bboxes.copy()
        return c

    def save(self, fname):
        data = self.to_list()
        with open(fname, "wb") as f:
            msgpack.dump(data, f, use_bin_type=True)

    @staticmethod
    def load(fname):
        with open(fname, "rb") as f:
            return CurveSet.from_list(msgpack.load(f, raw=False))

import surf
from functrace import trace

def align_down(n, to): return (n // to) * to
def align_up(n, to): return ((n + to - 1) // to) * to

class CurveRepainter:
    tile_width = 128
    tile_height = 128
    # one tile is <1% of the pixels in a full HD image
    def __init__(self, curve_set, curve_index, alpha_surface):
        self.curve_set = curve_set
        self.index = curve_index
        self.alpha_surface = alpha_surface
        # one thing we could do is render "the frame without the curve" when starting to edit
        # a line, and then repaint it on top of this frame every time. but this might involve
        # repainting all the lines, of which there might be many. so we amoritize the cost by
        # rendering the tiles necesssary
        w, h = alpha_surface.get_size()
        self.frame_without_curve = surf.AlphaSurface((w,h))
        self.rows = align_up(w, self.tile_width) // self.tile_width
        self.cols = align_up(h, self.tile_height) // self.tile_height

        self.painted = np.zeros((self.rows, self.cols), dtype=bool)

    def repaint(self, bbox):
        minx, miny, maxx, maxy = bbox
        w, h = self.alpha_surface.get_size()

        for y in range(align_down(miny, self.tile_height), align_up(maxy, self.tile_height), self.tile_height):
            for x in range(align_down(minx, self.tile_width), align_up(maxx, self.tile_width), self.tile_width):
                tile_row = x // self.tile_width
                tile_col = y // self.tile_height
                if not self.painted[tile_row, tile_col]:
                    trace.event('CurveRepainter-render-tiles')
                    paintWithin = np.array([x, y, min(x+self.tile_width, w), min(y+self.tile_height, h)], dtype=np.int32)
                    for i, curve in enumerate(self.curve_set.curves):
                        if i != self.index:
                            curve.render(self.frame_without_curve, paintWithin)
                    self.painted[tile_row, tile_col] = True

        self.alpha_surface.array()[minx:maxx,miny:maxy] = self.frame_without_curve.array()[minx:maxx,miny:maxy]

        paintWithin = np.array([minx, miny, maxx, maxy], dtype=np.int32)
        return self.curve_set.curves[self.index].render(self.alpha_surface, paintWithin, get_polyline=True)


