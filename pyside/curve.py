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
    return np.empty((n,4), dtype=np.int16)

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

    def render(self, alpha, paintWithin=None, get_polyline_of_index=None):
        polyline = None
        for i, curve in enumerate(self.curves):
            if paintWithin is not None and not rectangles_intersect(paintWithin, curve.pixels_bbox(self.bboxes[i])):
                continue # this curve is not affecting the paintWithin region
            curr_polyline = curve.render(alpha, paintWithin, get_polyline=get_polyline_of_index==i)
            if curr_polyline is not None:
                polyline = curr_polyline
        return polyline

    def closest_curve_index(self, x, y):
        # FIXME do it properly
        min_dist = 10**9
        min_ind = -1
        for i, (xmin, ymin, xmax, ymax) in enumerate(self.bboxes):
            xmid, ymid = (xmax-xmin)/2+xmin, (ymax-ymin)/2+ymin
            dist = math.sqrt((xmid-x)**2 + (ymid-y)**2)
            if dist < min_dist: 
                min_dist = dist
                min_ind = i
        return min_ind

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

