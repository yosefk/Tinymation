import math
import numpy as np
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

def polyline_array(n, dtype=float):
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
    arr[:,0] = maxval*np.round(polyline[:,0] - xmin)/(xmax - xmin)
    arr[:,1] = maxval*np.round(polyline[:,1] - ymin)/(ymax - ymin)
    arr[:,2] = maxval*polyline[:,2] # pressure is between 0 and 1

    return arr.tobytes(order='F'), bbox

def uncompress_polyline(polyline_bytes, bbox):
    polyline = np.frombuffer(polyline_bytes, dtype=np.uint16)
    polyline = polyline.reshape((len(polyline)//3, 3), order='F')

    arr = polyline_array(len(polyline))
    imaxval = 1/(2**16-1)

    arr[:,0] = imaxval*polyline*(xmax - xmin) + xmin
    arr[:,1] = imaxval*polyline*(ymax - ymin) + ymin
    arr[:,2] = imaxval*polyline[:,2]

    return arr, bbox

class Curve:
    def __init__(self, polyline, closed, brushConfig):
        self.polyline = polyline
        self.closed = closed
        self.brushConfig = brushConfig

    def to_dict(self, bbox=None):
        arr, bbox = compress_polyline(self.polyline, bbox)
        return dict(closed=self.closed, brush=self.brushConfig, bbox=bbox, polyline=arr)

    @staticmethod
    def from_dict(d):
        c = Curve(None,None,None)
        c.closed = d['closed']
        c.brushConfig = BrushConfig.from_dict(d['brush'])
        bbox = d['bbox']
        c.polyline = uncompress_polyline(d['polyline'], bbox)
        return c

    def copy(self):
        return Curve(self.polyline.copy(), self.closed, self.brushConfig.copy())

def bbox_array(n):
    return np.empty((n,4), dtype=np.int32)

class CurveSet:
    def __init__(self):
        self.curves = []
        # these are polyline points bboxes - expand each by its curve's maxPressureWidth brush setting plus a margin to get a bbox
        # of the points affected by the rendering of the polyline
        self.bboxes = bbox_array(0)
    def to_list(self):
        return [curve.to_dict(bbox) for (curve, bbox) in zip(self.curves,self.bboxes)]
    @staticmethod
    def from_list(ls):
        c = CurveSet()
        curves_and_bboxes = [Curve.from_dict(d) for d in ls]
        if not curves_and_bboxes:
            return c
        c.curves, bboxes = zip(*curves_and_bboxes)
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
        with open(fname, "wb") as f:
            msgpack.dump(self.to_list(), f, use_bin_type=True)

    @staticmethod
    def load(fname):
        with open(fname, "rb") as f:
            return CurveSet.from_list(msgpack.load(f, raw=False))

