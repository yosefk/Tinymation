import imageio
import imageio.v3
import numpy as np
import sys
import os

def compress_and_remove(filepairs):
    for infile, outfile in zip(filepairs[0::2], filepairs[1::2]):
        pixels = imageio.v3.imread(infile)
        imageio.imwrite(outfile, pixels)
        if np.array_equal(imageio.v3.imread(outfile), pixels):
            os.unlink(infile)

if len(sys.argv)>1 and sys.argv[1] == 'compress-and-remove':
    compress_and_remove(sys.argv[2:])
    exit()

import subprocess
import pygame
import pygame.gfxdraw
import winpath
import collections
import uuid
import math
import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')  # turn off interactive backend
import io
import json
import shutil
from scipy.interpolate import splprep, splev

# this requires numpy to be installed in addition to scikit-image
from skimage.morphology import flood_fill, binary_dilation, skeletonize
from scipy.ndimage import grey_dilation, grey_erosion, grey_opening, grey_closing
pg = pygame
pg.init()

#screen = pygame.display.set_mode((800, 350*2), pygame.RESIZABLE)
#screen = pygame.display.set_mode((350, 800), pygame.RESIZABLE)
#screen = pygame.display.set_mode((1200, 350), pygame.RESIZABLE)
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Tinymate")

IWIDTH = 1920
IHEIGHT = 1080
FRAME_RATE = 12
FADING_RATE = 3
PEN = (20, 20, 20)
BACKGROUND = (240, 235, 220)
MARGIN = (220, 215, 190)
UNDRAWABLE = (220-20, 215-20, 190-20)
SELECTED = (220-80, 215-80, 190-80)
LAYERS_BELOW = (0,128,255)
LAYERS_ABOVE = (255,128,0)
WIDTH = 3 # the smallest width where you always have a pure pen color rendered along
# the line path, making our naive flood fill work well...
MEDIUM_ERASER_WIDTH = 5*WIDTH
BIG_ERASER_WIDTH = 20*WIDTH
CURSOR_SIZE = int(screen.get_width() * 0.07)
MAX_HISTORY_BYTE_SIZE = 1*1024**3
MAX_CACHE_BYTE_SIZE = 1*1024**3
MAX_CACHED_ITEMS = 2000
FRAME_FMT = 'frame%04d.png'
CLIP_FILE = 'movie.json' # on Windows, this starting with 'm' while frame0000.png starts with 'f'
# makes the png the image inside the directory icon displayed in Explorer... which is very nice
FRAME_ORDER_FILE = 'frame_order.json'

MY_DOCUMENTS = winpath.get_my_documents()
WD = os.path.join(MY_DOCUMENTS if MY_DOCUMENTS else '.', 'Tinymate')
if not os.path.exists(WD):
    os.makedirs(WD)
print('clips read from, and saved to',WD)

import time
# add tdiff() to printouts to see how many ms passed since the last call to tdiff()
prevts=time.time_ns()
def tdiff():
    global prevts
    now=time.time_ns()
    diff=(now-prevts)//10**6
    prevts = now
    return diff

def image_from_fig(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im

fig = None
ax = None

def should_make_closed(curve_length, bbox_length, endpoints_dist):
    if curve_length < bbox_length*0.85:
        # if the distance between the endpoints is <30% of the length of the curve, close it
        return endpoints_dist / curve_length < 0.3
    else: # "long and curvy" - only make closed when the endpoints are close relatively to the bbox length
        return endpoints_dist / bbox_length < 0.1

def bspline_interp(points, suggest_options, existing_lines):
    x = np.array([1.*p[0] for p in points])
    y = np.array([1.*p[1] for p in points])

    okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
    x = np.r_[x[okay], x[-1]]#, x[0]]
    y = np.r_[y[okay], y[-1]]#, y[0]]

    def dist(i1, i2):
        return math.sqrt((x[i1]-x[i2])**2 + (y[i1]-y[i2])**2)
    curve_length = sum([dist(i, i+1) for i in range(len(x)-1)])

    results = []

    def add_result(tck, ufirst, ulast):
        step=(ulast-ufirst)/curve_length

        new_points = splev(np.arange(ufirst, ulast+step, step), tck)
        results.append(new_points)

    tck, u = splprep([x, y], s=len(x)/5)
    add_result(tck, u[0], u[-1])

    if not suggest_options:
        return results

    # check for intersections, throw out short segments between the endpoints and first/last intersection
    ix = np.round(results[0][0]).astype(int)
    iy = np.round(results[0][1]).astype(int)
    within_bounds = (ix >= 0) & (iy >= 0) & (ix < existing_lines.shape[0]) & (iy < existing_lines.shape[1])
    line_alphas = np.zeros(len(ix), int)
    line_alphas[within_bounds] = existing_lines[ix[within_bounds], iy[within_bounds]]
    intersections = np.where(line_alphas == 255)[0]

    def find_intersection_point(start, step):
        indexes = [intersections[start]]
        pos = start+step
        while pos < len(intersections) and pos >= 0 and abs(intersections[pos]-indexes[-1]) == 1:
            indexes.append(intersections[pos])
            pos += step
        return indexes[-1] #sum(indexes)/len(indexes)

    if len(intersections) > 0:
        len_first = intersections[0]
        len_last = len(ix) - intersections[-1]
        # look for clear alpha pixels along the path before the first and the last intersection - if we find some, we have >= 2 intersections
        two_or_more_intersections = len(np.where(line_alphas[intersections[0]:intersections[-1]] == 0)[0]) > 1

        first_short = two_or_more_intersections or len_first < len_last
        last_short = two_or_more_intersections or len_last <= len_first

        step=(u[-1]-u[0])/curve_length

        first_intersection = find_intersection_point(0, 1)
        last_intersection = find_intersection_point(len(intersections)-1, -1)
        uvals = np.arange(first_intersection if first_short else 0, (last_intersection if last_short else len(ix))+1, 1)*step
        new_points = splev(uvals, tck)
        return [new_points] + results

    # check if we'd like to attempt to close the line
    bbox_length = (np.max(x)-np.min(x))*2 + (np.max(y)-np.min(y))*2
    endpoints_dist = dist(0, -1)

    make_closed = len(points)>2 and should_make_closed(curve_length, bbox_length, endpoints_dist)

    if make_closed:
        tck, u = splprep([x, y], s=len(x)/5, per=True)
        add_result(tck, u[0], u[-1])
        return reversed(results)

    return results

def plotLines(points, ax, width, suggest_options, plot_reset, existing_lines):
    results = []
    def add_results(px, py):
        plot_reset()
        ax.plot(py,px, linestyle='solid', color='k', linewidth=width, scalex=False, scaley=False, solid_capstyle='round')
        results.append(image_from_fig(fig)[:,:,0])

    if len(set(points)) == 1:
        x,y = points[0]
        eps = 0.001
        points = [(x+eps, y+eps)] + points
    try:
        for path in bspline_interp(points, suggest_options, existing_lines):
            px, py = path[0], path[1]
            add_results(px, py)
    except:
        px = np.array([x for x,y in points])
        py = np.array([y for x,y in points])
        add_results(px, py)

    return results

def drawLines(image_height, image_width, points, width, suggest_options, existing_lines):
    global fig
    global ax
    if not fig:
        fig, ax = plt.subplots()
        ax.axis('off')
        fig.set_size_inches(image_width/fig.get_dpi(), image_height/fig.get_dpi())
        ok = False
        # not sure why it's needed, but for some w x h it is
        for epsx in np.arange(-0.01,0.02,0.01):
            for epsy in np.arange(-0.01,0.02,0.01):
                iff = image_from_fig(fig)
                if iff.shape == (image_height, image_width, 4):
                    ok = True
                    break
                fig.set_size_inches(image_width/fig.get_dpi()+epsx, image_height/fig.get_dpi()+epsy)
            if ok:
                break
        assert ok
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    width *= 72 / fig.get_dpi()

    def plot_reset():
        plt.cla()
        plt.xlim(0, image_width)
        plt.ylim(0, image_height)
        ax.invert_yaxis()
        ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
        ax.tick_params(left=False, right=False, bottom=False, top=False)

    return plotLines(points, ax, width, suggest_options, plot_reset, existing_lines)

def drawCircle( screen, x, y, color, width):
    pygame.draw.circle( screen, color, ( x, y ), width/2 )

def drawLine(screen, pos1, pos2, color, width):
    pygame.draw.line(screen, color, pos1, pos2, width)

def make_surface(width, height):
    return pg.Surface((width, height), screen.get_flags(), screen.get_bitsize(), screen.get_masks())

def scale_image(surface, width=None, height=None):
    assert width or height
    if not height:
        height = int(surface.get_height() * width / surface.get_width())
    if not width:
        width = int(surface.get_width() * height / surface.get_height())
    return pg.transform.smoothscale(surface, (width, height))

def minmax(v, minv, maxv):
    return min(maxv,max(minv,v))

def load_cursor(file, flip=False, size=CURSOR_SIZE, hot_spot=(0,1), min_alpha=192, edit=lambda x: x, hot_spot_offset=(0,0)):
  surface = pg.image.load(file)
  surface = scale_image(surface, size, size*surface.get_height()/surface.get_width())#pg.transform.scale(surface, (CURSOR_SIZE, CURSOR_SIZE))
  if flip:
      surface = pg.transform.flip(surface, True, True)
  non_transparent_surface = surface.copy()
  alpha = pg.surfarray.pixels_alpha(surface)
  alpha[:] = np.minimum(alpha, min_alpha)
  del alpha
  surface = edit(surface)
  hotx = minmax(int(hot_spot[0] * surface.get_width()) + hot_spot_offset[0], 0, surface.get_width()-1)
  hoty = minmax(int(hot_spot[1] * surface.get_height()) + hot_spot_offset[1], 0, surface.get_height()-1)
  return pg.cursors.Cursor((hotx, hoty), surface), non_transparent_surface

def add_circle(image, radius, color=(255,0,0,128), outline_color=(0,0,0,128)):
    new_width = radius + image.get_width()
    new_height = radius + image.get_height()
    result = pg.Surface((new_width, new_height), pg.SRCALPHA)
    pg.gfxdraw.filled_circle(result, radius, new_height-radius, radius, outline_color)
    pg.gfxdraw.filled_circle(result, radius, new_height-radius, radius-WIDTH+1, (0,0,0,0))
    pg.gfxdraw.filled_circle(result, radius, new_height-radius, radius-WIDTH+1, color)
    result.blit(image, (radius, 0))
    return result

pencil_cursor = load_cursor('pen.png')
pencil_cursor = (pencil_cursor[0], pg.image.load('pen-tool.png'))
eraser_cursor = load_cursor('eraser.png')
eraser_cursor = (eraser_cursor[0], pg.image.load('eraser-tool.png'))
eraser_medium_cursor = load_cursor('eraser.png', size=int(CURSOR_SIZE*1.5), edit=lambda s: add_circle(s, MEDIUM_ERASER_WIDTH//2), hot_spot_offset=(MEDIUM_ERASER_WIDTH//2,-MEDIUM_ERASER_WIDTH//2))
eraser_medium_cursor = (eraser_medium_cursor[0], eraser_cursor[1])
eraser_big_cursor = load_cursor('eraser.png', size=int(CURSOR_SIZE*2), edit=lambda s: add_circle(s, BIG_ERASER_WIDTH//2), hot_spot_offset=(BIG_ERASER_WIDTH//2,-BIG_ERASER_WIDTH//2))
eraser_big_cursor = (eraser_big_cursor[0], eraser_cursor[1])
flashlight_cursor = load_cursor('flashlight.png')
flashlight_cursor = (flashlight_cursor[0], pg.image.load('flashlight-tool.png')) 
paint_bucket_cursor = (load_cursor('paint_bucket.png')[1], pg.image.load('bucket-tool.png'))
blank_page_cursor = load_cursor('sheets.png', hot_spot=(0.5, 0.5))
garbage_bin_cursor = load_cursor('garbage.png', hot_spot=(0.5, 0.5))
# set_cursor can fail on some machines so we don't count on it to work.
# we set it early on to "give a sign of life" while the window is black;
# we reset it again before entering the event loop.
# if the cursors cannot be set the selected tool can still be inferred by
# the darker background of the tool selection button.
def try_set_cursor(c):
    try:
        pg.mouse.set_cursor(c)
    except:
        pass
try_set_cursor(pencil_cursor[0])

class CachedItem:
    def compute_key(self):
        '''a key is a tuple of:
        1. a list of tuples mapping IDs to versions. a cached item referencing
        unknown IDs or IDs with old versions is eventually garbage-collected
        2. any additional info making the key unique.
        
        compute_key returns the key computed from the current system state.
        for example, CachedThumbnail(pos=5) might return a dictionary mapping
        the IDs of frames making up frame 5 in every layer, and the string
        "thumbnail." The number 5 is not a part of the key; if frame 6
        is made of the same frames in every layer, CachedThumbnail(pos=6)
        will compute the same key. If in the future a CachedThumbnail(pos=5)
        is created computing a different key because the movie was edited,
        you'll get a cache miss as expected.
        '''
        return {}, None

    def compute_value(self):
        '''returns the value - used upon cached miss. note that CachedItems
        are not kept in the cache themselves - only the keys and the values.'''
        return None

# there are 2 reasons to evict a cached item:
# * no more room in the cache - evict the least recently used items until there's room
# * the cached item has no chance to be useful - eg it was computed from a deleted or
#   since-edited frame - this is done by collect_garbage() and assisted by update_id()
#   and delete_id()
class Cache:
    class Miss:
        pass
    MISS = Miss()
    def __init__(self):
        self.key2value = collections.OrderedDict()
        self.id2version = {}
        self.debug = False
        self.gc_iter = 0
        self.last_check = {}
        # these are per-gc iteration counters
        self.computed_bytes = 0
        self.cached_bytes = 0
        # sum([self.size(value) for value in self.key2value.values()])
        self.cache_size = 0
    def size(self,value):
        try:
            # surface
            return value.get_width() * value.get_height() * 4
        except:
            try:
                # numpy array
                return reduce(lambda x,y: x*y, value.shape)
            except:
                return 0
    def fetch(self, cached_item):
        key = cached_item.compute_key()
        value = self.key2value.get(key, Cache.MISS)
        if value is Cache.MISS:
            value = cached_item.compute_value()
            vsize = self.size(value)
            self.computed_bytes += vsize
            self.cache_size += vsize
            self._evict_lru_as_needed()
            self.key2value[key] = value
        else:
            self.key2value.move_to_end(key)
            self.cached_bytes += self.size(value)
            if self.debug and self.last_check.get(key, 0) < self.gc_iter:
                # slow debug mode
                ref = cached_item.compute_value()
                if not np.array_equal(pg.surfarray.pixels3d(ref), pg.surfarray.pixels3d(value)) or not np.array_equal(pg.surfarray.pixels_alpha(ref), pg.surfarray.pixels_alpha(value)):
                    print('HIT BUG!',key)
                self.last_check[key] = self.gc_iter
        return value

    def _evict_lru_as_needed(self):
        while self.cache_size > MAX_CACHE_BYTE_SIZE or len(self.key2value) > MAX_CACHED_ITEMS:
            key, value = self.key2value.popitem(last=False)
            self.cache_size -= self.size(value)

    def update_id(self, id, version):
        self.id2version[id] = version
    def delete_id(self, id):
        if id in self.id2version:
            del self.id2version[id]
    def stale(self, key):
        id2version, _ = key
        for id, version in id2version:
            current_version = self.id2version.get(id)
            if current_version is None or version < current_version:
                #print('stale',id,version,current_version)
                return True
        return False
    def collect_garbage(self):
        orig = len(self.key2value)
        orig_size = self.cache_size
        for key, value in list(self.key2value.items()):
            if self.stale(key):
                del self.key2value[key]
                self.cache_size -= self.size(value)
        #print('gc',orig,orig_size,'->',len(self.key2value),self.cache_size,'computed',self.computed_bytes,'cached',self.cached_bytes,tdiff())
        self.gc_iter += 1
        self.computed_bytes = 0
        self.cached_bytes = 0

cache = Cache()

def bounding_rectangle_of_a_boolean_mask(mask):
    # Sum along the vertical and horizontal axes
    vertical_sum = np.sum(mask, axis=1)
    if not np.any(vertical_sum):
        return None
    horizontal_sum = np.sum(mask, axis=0)

    minx, maxx = np.where(vertical_sum)[0][[0, -1]]
    miny, maxy = np.where(horizontal_sum)[0][[0, -1]]

    return minx, maxx, miny, maxy

class HistoryItem:
    def __init__(self, surface_id):
        self.surface_id = surface_id
        surface = self.curr_surface().copy()
        self.saved_alpha = pg.surfarray.pixels_alpha(surface)
        self.saved_rgb = pg.surfarray.pixels3d(surface) if surface_id == 'color' else None
        self.pos = movie.pos
        self.layer_pos = movie.layer_pos
        self.minx = 10**9
        self.miny = 10**9
        self.maxx = -10**9
        self.maxy = -10**9
        self.optimized = False
    def curr_surface(self):
        return movie.edit_curr_frame().surf_by_id(self.surface_id)
    def nop(self):
        return self.saved_alpha is None
    def undo(self):
        if self.nop():
            return

        if self.pos != movie.pos or self.layer_pos != movie.layer_pos:
            print(f'WARNING: HistoryItem at the wrong position! should be {self.pos} [layer {self.layer_pos}], but is {movie.pos} [layer {movie.layer_pos}]')
        movie.seek_frame_and_layer(self.pos, self.layer_pos) # we should already be here, but just in case

        # we could have created this item a bit more quickly with a bit more code but doesn't seem worth it
        redo = HistoryItem(self.surface_id)

        frame = self.curr_surface()
        if self.optimized:
            pg.surfarray.pixels_alpha(frame)[self.minx:self.maxx+1, self.miny:self.maxy+1] = self.saved_alpha
            if self.saved_rgb is not None:
                pg.surfarray.pixels3d(frame)[self.minx:self.maxx+1, self.miny:self.maxy+1] = self.saved_rgb
        else:
            pg.surfarray.pixels_alpha(frame)[:] = self.saved_alpha
            if self.saved_rgb is not None:
                pg.surfarray.pixels3d(frame)[:] = self.saved_rgb

        redo.optimize()
        return redo
    def optimize(self):
        mask = self.saved_alpha != pg.surfarray.pixels_alpha(self.curr_surface())
        if self.saved_rgb is not None:
            mask |= np.any(self.saved_rgb != pg.surfarray.pixels3d(self.curr_surface()), axis=2)
        brect = bounding_rectangle_of_a_boolean_mask(mask)

        if brect is None: # this can happen eg when drawing lines on an already-filled-with-lines area
            self.saved_alpha = None
            self.saved_rgb = None
            return
        
        self.minx, self.maxx, self.miny, self.maxy = brect
        self.saved_alpha = self.saved_alpha[self.minx:self.maxx+1, self.miny:self.maxy+1].copy()
        if self.saved_rgb is not None:
            self.saved_rgb = self.saved_rgb[self.minx:self.maxx+1, self.miny:self.maxy+1].copy()
        self.optimized = True

    def __str__(self):
        return f'HistoryItem(pos={self.pos}, rect=({self.minx}, {self.miny}, {self.maxx}, {self.maxy}))'

    def byte_size(self):
        if self.nop():
            return 0
        return self.saved_alpha.nbytes + (self.saved_rgb.nbytes if self.saved_rgb is not None else 0)

class HistoryItemSet:
    def __init__(self, items):
        self.items = [item for item in items if item is not None]
    def nop(self):
        for item in self.items:
            if not item.nop():
                return False
        return True
    def undo(self):
        return HistoryItemSet(list(reversed([item.undo() for item in self.items])))
    def optimize(self):
        for item in self.items:
            item.optimize()
        self.items = [item for item in self.items if not item.nop()]
    def byte_size(self):
        return sum([item.byte_size() for item in self.items])

def scale_and_preserve_aspect_ratio(w, h, width, height):
    if width/height > w/h:
        scaled_width = w*height/h
        scaled_height = h*scaled_width/w
    else:
        scaled_height = h*width/w
        scaled_width = w*scaled_height/h
    return scaled_width, scaled_height

class Button:
    def __init__(self):
        self.button_surface = None
    def draw(self, rect, cursor_surface):
        left, bottom, width, height = rect
        _, _, w, h = cursor_surface.get_rect()
        scaled_width, scaled_height = scale_and_preserve_aspect_ratio(w, h, width, height)
        if not self.button_surface:
            surface = scale_image(cursor_surface, scaled_width, scaled_height)
            self.button_surface = surface
        screen.blit(self.button_surface, (left+(width-scaled_width)/2, bottom+height-scaled_height))

locked_image = pg.image.load('locked.png')
invisible_image = pg.image.load('eye_shut.png')
def curr_layer_locked():
    effectively_locked = movie.curr_layer().locked or not movie.curr_layer().visible
    if effectively_locked: # invisible layers are effectively locked but we show it differently
        reason_image = locked_image if movie.curr_layer().locked else invisible_image
        fading_mask = layout.drawing_area().new_frame()
        fading_mask.blit(reason_image, ((fading_mask.get_width()-reason_image.get_width())//2, (fading_mask.get_height()-reason_image.get_height())//2))
        fading_mask.set_alpha(192)
        layout.drawing_area().set_fading_mask(fading_mask)
        layout.drawing_area().fade_per_frame = 192/(FADING_RATE*3)
    return effectively_locked

class PenTool(Button):
    def __init__(self, eraser=False, width=WIDTH):
        Button.__init__(self)
        self.prev_drawn = None
        self.color = BACKGROUND if eraser else PEN
        self.eraser = eraser
        self.width = width
        self.circle_width = (width//2)*2
        self.points = []
        self.lines_array = None
        self.suggestion_mask = None
        if self.eraser:
            self.alpha_surface = None

    def on_mouse_down(self, x, y):
        if curr_layer_locked():
            return
        self.points = []
        self.bucket_color = None
        self.lines_array = pg.surfarray.pixels_alpha(movie.edit_curr_frame().surf_by_id('lines'))
        if self.eraser:
            if not self.alpha_surface:
                self.alpha_surface = layout.drawing_area().new_frame()
            pg.surfarray.pixels_red(self.alpha_surface)[:] = 0
        self.on_mouse_move(x,y)

    def on_mouse_up(self, x, y):
        if curr_layer_locked():
            return
        self.lines_array = None
        drawing_area = layout.drawing_area()
        cx, cy = drawing_area.xy2frame(x, y)
        self.points.append((cx,cy))
        self.prev_drawn = None
        frame = movie.edit_curr_frame().surf_by_id('lines')
        lines = pygame.surfarray.pixels_alpha(frame)

        prev_history_item = None
        line_width = self.width * (1 if self.width == WIDTH else drawing_area.xscale)
        line_options = drawLines(frame.get_width(), frame.get_height(), self.points, line_width, suggest_options=not self.eraser, existing_lines=lines)
        items = []
        for new_lines in line_options:
            history_item = HistoryItem('lines')
            if prev_history_item:
                prev_history_item.undo()
            if self.eraser:
                lines[:] = np.minimum(new_lines, lines)
            else:
                lines[:] = np.maximum(255-new_lines, lines)

            if self.eraser:
                color_history_item = HistoryItem('color')
                color = movie.edit_curr_frame().surf_by_id('color')
                color_rgb = pg.surfarray.pixels3d(color)
                color_alpha = pg.surfarray.pixels_alpha(color)
                flood_fill_color_based_on_lines(color_rgb, color_alpha, lines, round(cx), round(cy), self.bucket_color if self.bucket_color else BACKGROUND+(0,))
                history_item = HistoryItemSet([history_item, color_history_item])

            history_item.optimize()
            items.append(history_item)
            if not prev_history_item:
                prev_history_item = history_item

        history.append_suggestions(items)
        
        if len(line_options)>1:
            if self.suggestion_mask is None:
                left, bottom, width, height = drawing_area.rect
                self.suggestion_mask = pg.Surface((IWIDTH, IHEIGHT), pg.SRCALPHA)
                self.suggestion_mask.fill((0,255,0))
            alt_option = line_options[-2]
            pg.surfarray.pixels_alpha(self.suggestion_mask)[:] = 255-alt_option
            self.suggestion_mask.set_alpha(10)
            drawing_area.set_fading_mask(self.suggestion_mask)
            class Fading:
                def __init__(self):
                    self.i = 0
                def fade(self, alpha, _):
                    self.i += 1
                    if self.i == 1:
                        return 10
                    if self.i == 2:
                        return 130
                    else:
                        return 110-self.i*10
            drawing_area.fading_func = Fading().fade

    def on_mouse_move(self, x, y):
        if curr_layer_locked():
            return
        drawing_area = layout.drawing_area()
        cx, cy = drawing_area.xy2frame(x, y)
        if self.eraser and self.bucket_color is None:
            nx, ny = round(cx), round(cy)
            if nx>=0 and ny>=0 and nx<self.lines_array.shape[0] and ny<self.lines_array.shape[1] and self.lines_array[nx,ny] == 0:
                self.bucket_color = movie.edit_curr_frame().surf_by_id('color').get_at((cx,cy))
        self.points.append((cx,cy))
        color = self.color if not self.eraser else (self.bucket_color if self.bucket_color else (255,255,255,0))
        expose_other_layers = self.eraser and color[3]==0
        if expose_other_layers:
            color = (255,0,0,0)
        draw_into = drawing_area.subsurface if not expose_other_layers else self.alpha_surface
        ox,oy = (0,0) if not expose_other_layers else (drawing_area.xmargin, drawing_area.ymargin)
        if self.prev_drawn:
            drawLine(draw_into, (self.prev_drawn[0]-ox, self.prev_drawn[1]-oy), (x-ox,y-oy), color, self.width)
        drawCircle(draw_into, x-ox, y-oy, color, self.circle_width)
        if expose_other_layers:
            alpha = pg.surfarray.pixels_red(draw_into)
            w, h = self.lines_array.shape
            def clipw(val): return max(0, min(val, w))
            def cliph(val): return max(0, min(val, h))
            px, py = self.prev_drawn if self.prev_drawn else (x, y)
            left = clipw(min(x-ox-self.width, px-ox-self.width))
            right = clipw(max(x-ox+self.width, px-ox+self.width))
            bottom = cliph(min(y-oy-self.width, py-oy-self.width))
            top = cliph(max(y-oy+self.width, py-oy+self.width))
            def render_surface(s):
                if not s:
                    return
                salpha = pg.surfarray.pixels_alpha(s)
                orig_alpha = salpha[left:right+1, bottom:top+1].copy()
                salpha[left:right+1, bottom:top+1] = np.minimum(orig_alpha, alpha[left:right+1,bottom:top+1])
                del salpha
                drawing_area.subsurface.blit(s, (ox+left,oy+bottom), (left,bottom,right-left+1,top-bottom+1))
                salpha = pg.surfarray.pixels_alpha(s)
                salpha[left:right+1, bottom:top+1] = orig_alpha

            render_surface(movie.curr_bottom_layers_surface(movie.pos, highlight=True))
            render_surface(movie.curr_top_layers_surface(movie.pos, highlight=True))
            render_surface(layout.timeline_area().combined_light_table_mask())

        self.prev_drawn = (x,y) 

class NewDeleteTool(PenTool):
    def __init__(self, frame_func, clip_func, layer_func):
        PenTool.__init__(self)
        self.frame_func = frame_func
        self.clip_func = clip_func
        self.layer_func = layer_func

    def on_mouse_down(self, x, y): pass
    def on_mouse_up(self, x, y): pass
    def on_mouse_move(self, x, y): pass

def flood_fill_color_based_on_lines(color_rgb, color_alpha, lines, x, y, bucket_color):
    pen_mask = lines == 255
    flood_code = 2
    flood_mask = flood_fill(pen_mask.astype(np.byte), (x,y), flood_code) == flood_code
    for ch in range(3):
         color_rgb[:,:,ch] = color_rgb[:,:,ch]*(1-flood_mask) + bucket_color[ch]*flood_mask
    color_alpha[:] = color_alpha*(1-flood_mask) + bucket_color[3]*flood_mask

class PaintBucketTool(Button):
    def __init__(self,color):
        Button.__init__(self)
        self.color = color
    def on_mouse_down(self, x, y):
        if curr_layer_locked():
            return
        x, y = layout.drawing_area().xy2frame(x,y)
        x, y = round(x), round(y)
        color_surface = movie.edit_curr_frame().surf_by_id('color')
        color_rgb = pg.surfarray.pixels3d(color_surface)
        color_alpha = pg.surfarray.pixels_alpha(color_surface)
        lines = pygame.surfarray.pixels_alpha(movie.edit_curr_frame().surf_by_id('lines'))

        if x < 0 or y < 0 or x >= lines.shape[0] or y >= lines.shape[1]:
            return
        
        if (np.array_equal(color_rgb[x,y,:], np.array(self.color[0:3])) and color_alpha[x,y] == self.color[3]) or lines[x,y] == 255:
            return # we never flood the lines themselves - they keep the PEN color in a separate layer;
            # and there's no point in flooding with the color the pixel already has

        history_item = HistoryItem('color')

        flood_fill_color_based_on_lines(color_rgb, color_alpha, lines, x, y, self.color)

        history_item.optimize()
        history.append_item(history_item)
        
    def on_mouse_up(self, x, y):
        pass
    def on_mouse_move(self, x, y):
        pass

NO_PATH_DIST = 10**6

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def skeleton_to_distances(skeleton, x, y):
    width, height = skeleton.shape
    yg, xg = np.meshgrid(np.arange(height), np.arange(width))
    dist = np.sqrt((xg - x)**2 + (yg - y)**2)

    # look at points around the point on the skeleton closest to the selected point
    # (and not around the selected point itself since nothing could be close enough to it)
    closest = np.argmin(dist[skeleton])
    x,y = xg[skeleton].flat[closest],yg[skeleton].flat[closest]

    dist = np.sqrt((xg - x)**2 + (yg - y)**2)
    skx, sky = np.where(skeleton & (dist < 200))
    if len(skx) == 0:
        return np.ones((width, height), int) * NO_PATH_DIST, NO_PATH_DIST
    closest = np.argmin((skx-x)**2 + (sky-y)**2)

    ixy = list(enumerate(zip(skx,sky)))
    xy2i = dict([((x,y),i) for i,(x,y) in ixy])

    data = [] 
    row_ind = []
    col_ind = []

    width, height = skeleton.shape
    neighbors = [(ox, oy) for ox in range(-1,2) for oy in range(-1,2) if ox or oy]
    for i,(x,y) in ixy:
        for ox, oy in neighbors:
            nx = ox+x
            ny = oy+y
            if nx >= 0 and ny >= 0 and nx < width and ny < height:
                j = xy2i.get((nx,ny), None)
                if j is not None:
                    data.append(1)
                    row_ind.append(i)
                    col_ind.append(xy2i[(nx,ny)])
    
    graph = csr_matrix((data, (row_ind, col_ind)), (len(ixy), len(ixy)))
    distance_matrix = dijkstra(graph, directed=False)

    distances = np.ones((width, height), int) * NO_PATH_DIST
    maxdist = 0
    for i,(x,y) in ixy:
        d = distance_matrix[closest,i]
        if not math.isinf(d):
            distances[x,y] = d
            maxdist = max(maxdist, d)

    return distances, maxdist

last_flood_mask = None
last_skeleton = None

import colorsys

def skeletonize_color_based_on_lines(color, lines, x, y):
    global last_flood_mask
    global last_skeleton

    pen_mask = lines == 255
    if pen_mask[x,y]:
        return

    flood_code = 2
    flood_mask = flood_fill(pen_mask.astype(np.byte), (x,y), flood_code) == flood_code
    if last_flood_mask is not None and np.array_equal(flood_mask, last_flood_mask):
        skeleton = last_skeleton
    else: 
        skeleton = skeletonize(flood_mask)
        last_flood_mask = flood_mask
        last_skeleton = skeleton

    fmb = binary_dilation(binary_dilation(skeleton))
    fading_mask = pg.Surface((flood_mask.shape[0], flood_mask.shape[1]), pg.SRCALPHA)
    fm = pg.surfarray.pixels3d(fading_mask)
    yg, xg = np.meshgrid(np.arange(flood_mask.shape[1]), np.arange(flood_mask.shape[0]))

    # Compute distance from each point to the specified center
    d, maxdist = skeleton_to_distances(skeleton, x, y)
    if maxdist != NO_PATH_DIST:
        d = (d == NO_PATH_DIST)*maxdist + (d != NO_PATH_DIST)*d # replace NO_PATH_DIST with maxdist
    else: # if all the pixels are far from clicked coordinate, make the mask bright instead of dim,
        # otherwise it might look like "the flashlight isn't working"
        #
        # note that this case shouldn't happen because we are highlighting points around the closest
        # point on the skeleton to the clocked coordinate and not around the clicked coordinate itself
        d = np.ones(lines.shape, int)
        maxdist = 10
    outer_d = -grey_dilation(-d, 3)
    inner = (255,255,255)
    outer = [255-ch for ch in color[x,y]]
    h,s,v = colorsys.rgb_to_hsv(*[o/255. for o in outer])
    s = 1
    v = 1
    outer = [255*o for o in colorsys.hsv_to_rgb(h,s,v)]
    for ch in range(3):
         fm[:,:,ch] = outer[ch]*(1-skeleton) + inner[ch]*skeleton
    pg.surfarray.pixels_alpha(fading_mask)[:] = fmb*255*np.maximum(0,(1- .90*outer_d/maxdist))

    return fading_mask

class FlashlightTool(Button):
    def __init__(self):
        Button.__init__(self)
    def on_mouse_down(self, x, y):
        x, y = layout.drawing_area().xy2frame(x,y)
        x, y = round(x), round(y)
        color = pygame.surfarray.pixels3d(movie.curr_frame().surf_by_id('color'))
        lines = pygame.surfarray.pixels_alpha(movie.curr_frame().surf_by_id('lines'))
        if x >= color.shape[0] or y >= color.shape[1]:
            return
        fading_mask = skeletonize_color_based_on_lines(color, lines, x, y)
        if not fading_mask:
            return
        fading_mask.set_alpha(255)
        layout.drawing_area().set_fading_mask(fading_mask)
        layout.drawing_area().fade_per_frame = 255/(FADING_RATE*15)
    def on_mouse_up(self, x, y): pass
    def on_mouse_move(self, x, y): pass

# layout:
#
# - some items can change the cursor [specifically the timeline], so need to know to restore it back to the
#   "current default cursor" when it was changed from it and the current mouse position is outside the
#   "special cursor area"
#
# - some items can change the current tool [specifically tool selection buttons], which changes the
#   current default cursor too 
#
# - the drawing area makes use of the current tool
#
# - the element sizes are relative to the screen size. [within its element area, the drawing area
#   and the timeline images use a 16:9 subset]

class Layout:
    def __init__(self):
        self.elems = []
        _, _, self.width, self.height = screen.get_rect()
        self.is_pressed = False
        self.is_playing = False
        self.playing_index = 0
        self.tool = PenTool()
        self.full_tool = TOOLS['pencil']
        self.focus_elem = None

    def aspect_ratio(self): return self.width/self.height

    def add(self, rect, elem):
        left, bottom, width, height = rect
        srect = (round(left*self.width), round(bottom*self.height), round(width*self.width), round(height*self.height))
        elem.rect = srect
        elem.subsurface = screen.subsurface(srect)
        getattr(elem, 'init', lambda: None)()
        self.elems.append(elem)

    def draw(self):
        if self.is_pressed and self.focus_elem is self.drawing_area():
            return
        screen.fill(MARGIN if self.is_playing else UNDRAWABLE)
        for elem in self.elems:
            if not self.is_playing or isinstance(elem, DrawingArea) or isinstance(elem, TogglePlaybackButton):
                elem.draw()
            #pygame.draw.rect(screen, PEN, elem.rect, 1, 1)

    # note that pygame seems to miss mousemove events with a Wacom pen when it's not pressed.
    # (not sure if entirely consistently.) no such issue with a regular mouse
    def on_event(self,event):
        if event.type == PLAYBACK_TIMER_EVENT:
            if self.is_playing:
                self.playing_index = (self.playing_index + 1) % len(movie.frames)

        if event.type == FADING_TIMER_EVENT:
            self.drawing_area().update_fading_mask()

        if event.type == SAVING_TIMER_EVENT:
            movie.frame(movie.pos).save()

        if event.type not in [pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION]:
            return

        x, y = pygame.mouse.get_pos()

        dispatched = False
        for elem in self.elems:
            left, bottom, width, height = elem.rect
            if x>=left and x<left+width and y>=bottom and y<bottom+height:
                if not self.is_playing or isinstance(elem, TogglePlaybackButton):
                    self._dispatch_event(elem, event, x, y)
                    dispatched = True
                    break

        if not dispatched and self.focus_elem:
            self._dispatch_event(None, event, x, y)
            return


    def _dispatch_event(self, elem, event, x, y):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.is_pressed = True
            self.focus_elem = elem
            self.focus_elem.on_mouse_down(x,y)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_pressed = False
            if self.focus_elem:
                self.focus_elem.on_mouse_up(x,y)
            self.focus_elem = None
        elif event.type == pygame.MOUSEMOTION and self.is_pressed:
            if self.focus_elem:
                self.focus_elem.on_mouse_move(x,y)

    def drawing_area(self):
        assert isinstance(self.elems[0], DrawingArea)
        return self.elems[0]

    def timeline_area(self):
        assert isinstance(self.elems[1], TimelineArea)
        return self.elems[1]

    def movie_list_area(self):
        assert isinstance(self.elems[2], MovieListArea)
        return self.elems[2]

    def toggle_playing(self):
        self.is_playing = not self.is_playing
        self.playing_index = 0
            
class DrawingArea:
    def __init__(self):
        self.fading_mask = None
        self.fading_func = None
        self.fade_per_frame = 0
        self.last_update_time = 0
        self.ymargin = WIDTH * 3
        self.xmargin = WIDTH * 3
        self.render_surface = None
        self.iwidth = 0
        self.iheight = 0
    def _internal_layout(self):
        if self.iwidth and self.iheight:
            return
        left, bottom, width, height = self.rect
        self.iwidth, self.iheight = scale_and_preserve_aspect_ratio(IWIDTH, IHEIGHT, width - self.xmargin*2, height - self.ymargin*2)
        self.xmargin = round((width - self.iwidth)/2)
        self.ymargin = round((height - self.iheight)/2)
        self.xscale = IWIDTH/self.iwidth
        self.yscale = IHEIGHT/self.iheight
    def xy2frame(self, x, y):
        return (x - self.xmargin)*self.xscale, (y - self.ymargin)*self.yscale
    def scale(self, surface): return scale_image(surface, self.iwidth, self.iheight)
    def set_fading_mask(self, fading_mask): self.fading_mask = self.scale(fading_mask)
    def draw(self):
        self._internal_layout()
        left, bottom, width, height = self.rect
        if not layout.is_playing:
            pygame.draw.rect(self.subsurface, MARGIN, (0, 0, width, self.ymargin))
            pygame.draw.rect(self.subsurface, MARGIN, (0, 0, self.xmargin, height))
            pygame.draw.rect(self.subsurface, MARGIN, (width-self.xmargin, 0, self.xmargin, height))
            pygame.draw.rect(self.subsurface, MARGIN, (0, height-self.ymargin, width, self.ymargin))

        pos = layout.playing_index if layout.is_playing else movie.pos
        highlight = not layout.is_playing and not movie.curr_layer().locked
        starting_point = (self.xmargin, self.ymargin)
        self.subsurface.blit(movie.curr_bottom_layers_surface(pos, highlight=highlight), starting_point)
        if movie.layers[movie.layer_pos].visible:
            frame = movie.frame(pos).surface()
            self.subsurface.blit(self.scale(frame), starting_point)
        self.subsurface.blit(movie.curr_top_layers_surface(pos, highlight=highlight), starting_point)

        if not layout.is_playing:
            mask = layout.timeline_area().combined_light_table_mask()
            if mask:
                self.subsurface.blit(mask, starting_point)
            if self.fading_mask:
                self.subsurface.blit(self.fading_mask, starting_point)

    def update_fading_mask(self):
        if not self.fading_mask:
            return
        now = time.time_ns()
        ignore_event = (now - self.last_update_time) // 10**6 < (1000 / (FRAME_RATE*2))
        self.last_update_time = now

        if ignore_event:
            return

        alpha = self.fading_mask.get_alpha()
        if alpha == 0:
            self.fading_mask = None
            self.fading_func = None
            return

        if not self.fading_func:
            alpha -= self.fade_per_frame
        else:
            alpha = self.fading_func(alpha, self.fade_per_frame)
        self.fading_mask.set_alpha(max(0,alpha))

    def fix_xy(self,x,y):
        left, bottom, _, _ = self.rect
        return (x-left), (y-bottom)
    def on_mouse_down(self,x,y):
        layout.tool.on_mouse_down(*self.fix_xy(x,y))
    def on_mouse_up(self,x,y):
        left, bottom, _, _ = self.rect
        layout.tool.on_mouse_up(*self.fix_xy(x,y))
    def on_mouse_move(self,x,y):
        left, bottom, _, _ = self.rect
        layout.tool.on_mouse_move(*self.fix_xy(x,y))
    def new_frame(self):
        _, _, width, height = self.rect
        frame = pygame.Surface((IWIDTH, IHEIGHT), pygame.SRCALPHA)
        frame.fill(BACKGROUND)
        pg.surfarray.pixels_alpha(frame)[:] = 0
        return frame

class TimelineArea:
    def _calc_factors(self):
        # TODO: handle layout so narrow that you have to scale the middle thumbnail
        factors = [0.7,0.6,0.5,0.4,0.3,0.2,0.15]
        scale = 1
        mid_scale = 1
        step = 0.5
        mid_width = IWIDTH * screen.get_height() * 0.15 / IHEIGHT
        def scaled_factors(scale):
            return [min(1, max(0.15, f*scale)) for f in factors]
        def slack(scale):
            total_width = mid_width*mid_scale + 2 * sum([int(mid_width)*f for f in scaled_factors(scale)])
            return screen.get_width() - total_width
        prev_slack = None
        iteration = 0
        while iteration < 1000:
            opt = [scale+step, scale-step, scale+step/2, scale-step/2]
            slacks = [abs(slack(s)) for s in opt]
            best_slack = min(slacks)
            best_opt = opt[slacks.index(best_slack)]

            step = best_opt - scale
            scale = best_opt

            curr_slack = slack(scale)
            def nice_fit(): return curr_slack >= 0 and curr_slack < 2
            if nice_fit():
                break
            
            sf = scaled_factors(scale)
            if min(sf) == 1: # grown as big as we will allow?
                break

            if max(sf) == 0.15: # grown as small as we will allow? try shrinking the middle thumbnail
                while not nice_fit() and mid_scale > 0.15:
                    mid_scale = max(scale-0.1, 0.15)
                    curr_slack = slack(scale)
                break # can't do much if we still don't have a nice fit

            iteration += 1
            
        self.factors = scaled_factors(scale)
        self.mid_factor = mid_scale

    def __init__(self):
        # stuff for drawing the timeline
        self.frame_boundaries = []
        self.eye_boundaries = []
        self.prevx = None

        self._calc_factors()

        eye_icon_size = int(screen.get_width() * 0.15*0.14)
        self.eye_open = scale_image(pg.image.load('light_on.png'), eye_icon_size)
        self.eye_shut = scale_image(pg.image.load('light_off.png'), eye_icon_size)

        self.loop_icon = scale_image(pg.image.load('loop.png'), int(screen.get_width()*0.15*0.14))
        self.arrow_icon = scale_image(pg.image.load('arrow.png'), int(screen.get_width()*0.15*0.2))

        self.no_hold = scale_image(pg.image.load('no_hold.png'), int(screen.get_width()*0.15*0.25))
        self.hold_active = scale_image(pg.image.load('hold_yellow.png'), int(screen.get_width()*0.15*0.25))
        self.hold_inactive = scale_image(pg.image.load('hold_grey.png'), int(screen.get_width()*0.15*0.25))

        # stuff for light table [what positions are enabled and what the resulting
        # mask to be rendered together with the current frame is]
        self.on_light_table = {}
        for pos_dist in range(-len(self.factors),len(self.factors)+1):
            self.on_light_table[pos_dist] = False
        self.on_light_table[-1] = True
        # the order in which we traverse the masks matters, for one thing,
        # because we might cover the same position distance from movie.pos twice
        # due to wraparound, and we want to decide if it's covered as being
        # "before" or "after" movie pos [it affects the mask color]
        self.traversal_order = []
        for pos_dist in range(1,len(self.factors)+1):
            self.traversal_order.append(-pos_dist)
            self.traversal_order.append(pos_dist)

        self.loop_mode = False

        self.toggle_hold_boundaries = (0,0,0,0)
        self.loop_boundaries = (0,0,0,0)

    def light_table_positions(self):
        # TODO: order 
        covered_positions = {movie.pos} # the current position is definitely covered,
        # don't paint over it...

        num_enabled_pos = sum([enabled for pos_dist, enabled in self.on_light_table.items() if pos_dist>0])
        num_enabled_neg = sum([enabled for pos_dist, enabled in self.on_light_table.items() if pos_dist<0])
        curr_pos = 0
        curr_neg = 0
        for pos_dist in self.traversal_order:
            if not self.on_light_table[pos_dist]:
                continue
            abs_pos = movie.pos + pos_dist
            if not self.loop_mode and (abs_pos < 0 or abs_pos >= len(movie.frames)):
                continue
            pos = abs_pos % len(movie.frames)
            if pos in covered_positions:
                continue # for short movies, avoid covering the same position twice
                # upon wraparound
            covered_positions.add(pos)
            if pos_dist > 0:
                curr = curr_pos
                num = num_enabled_pos
                curr_pos += 1
            else:
                curr = curr_neg
                num = num_enabled_neg
                curr_neg += 1
            brightness = int((200 * (num - curr - 1) / (num - 1)) + 55 if num > 1 else 255)
            color = (0,0,brightness) if pos_dist < 0 else (0,int(brightness*0.5),0)
            transparency = 0.3
            yield (pos, color, transparency)

    def combined_light_table_mask(self):
        class CachedCombinedMask:
            def compute_key(_):
                id2version = []
                computation = []
                for pos, color, transparency in self.light_table_positions():
                    i2v, c = movie.get_mask(pos, color, transparency, key=True)
                    id2version += i2v
                    computation.append(c)
                return tuple(id2version), ('combined-mask', tuple(computation))
                
            def compute_value(_):
                masks = []
                for pos, color, transparency in self.light_table_positions():
                    masks.append(movie.get_mask(pos, color, transparency))
                scale = layout.drawing_area().scale
                if len(masks) == 0:
                    return None
                elif len(masks) == 1:
                    return scale(masks[0])
                else:
                    mask = masks[0].copy()
                    alphas = []
                    for m in masks[1:]:
                        alphas.append(m.get_alpha())
                        m.set_alpha(255) # TODO: this assumes the same transparency in all masks - might want to change
                    mask.blits([(m, (0, 0), (0, 0, mask.get_width(), mask.get_height())) for m in masks[1:]])
                    for m,a in zip(masks[1:],alphas):
                        m.set_alpha(a)
                    return scale(mask)

        return cache.fetch(CachedCombinedMask())

    def x2frame(self, x):
        for left, right, pos in self.frame_boundaries:
            if x >= left and x <= right:
                return pos
    def draw(self):
        left, bottom, width, height = self.rect
        frame_width = movie.curr_frame().get_width()
        frame_height = movie.curr_frame().get_height()
        #thumb_width = movie.curr_frame().get_width() * height // movie.curr_frame().get_height()
        x = left
        i = 0

        factors = self.factors
        self.frame_boundaries = []
        self.eye_boundaries = []

        def draw_frame(pos, pos_dist, x, thumb_width):
            scaled = movie.get_thumbnail(pos, thumb_width, height)
            screen.blit(scaled, (x, bottom), (0, 0, thumb_width, height))
            border = 1 + 2*(pos==movie.pos)
            pygame.draw.rect(screen, PEN, (x, bottom, thumb_width, height), border)
            self.frame_boundaries.append((x, x+thumb_width, pos))
            if pos != movie.pos:
                eye = self.eye_open if self.on_light_table.get(pos_dist, False) else self.eye_shut
                eye_x = x + 2 if pos_dist > 0 else x+thumb_width-eye.get_width() - 2
                screen.blit(eye, (eye_x, bottom), eye.get_rect())
                self.eye_boundaries.append((eye_x, bottom, eye_x+eye.get_width(), bottom+eye.get_height(), pos_dist))
            elif len(movie.frames)>1:
                mode_x = x + 2
                mode = self.loop_icon if self.loop_mode else self.arrow_icon
                screen.blit(mode, (mode_x, bottom), mode.get_rect())
                self.loop_boundaries = (mode_x, bottom, mode_x+mode.get_width(), bottom+mode.get_height())

        def thumb_width(factor):
            return int((frame_width * height // frame_height) * factor)

        # current frame
        curr_frame_width = thumb_width(self.mid_factor)
        centerx = (left+width)/2
        draw_frame(movie.pos, 0, centerx - curr_frame_width/2, curr_frame_width)

        # next frames
        x = centerx + curr_frame_width/2
        i = 0
        pos = movie.pos + 1
        while True:
            if i >= len(factors):
                break
            if not self.loop_mode and pos >= len(movie.frames):
                break
            if pos >= len(movie.frames): # went past the last frame
                pos = 0
            if pos == movie.pos: # gone all the way back to the current frame
                break
            ith_frame_width = thumb_width(factors[i])
            draw_frame(pos, i+1, x, ith_frame_width)
            x += ith_frame_width
            pos += 1
            i += 1

        # previous frames
        x = centerx - curr_frame_width/2
        i = 0
        pos = movie.pos - 1
        while True:
            if i >= len(factors):
                break
            if not self.loop_mode and pos < 0:
                break
            if pos < 0: # went past the first frame
                pos = len(movie.frames) - 1
            if pos == movie.pos: # gone all the way back to the current frame
                break
            ith_frame_width = thumb_width(factors[i])
            x -= ith_frame_width
            draw_frame(pos, -i-1, x, ith_frame_width)
            pos -= 1
            i += 1

        self.draw_hold()

    def draw_hold(self):
        left, bottom, width, height = self.rect
        # sort by position for nicer looking occlusion between adjacent icons
        for left, right, pos in sorted(self.frame_boundaries, key=lambda x: x[2]):
            if pos == 0:
                continue # can't toggle hold at frame 0
            if movie.frames[pos].hold:
                hold = self.hold_active if pos == movie.pos else self.hold_inactive
            elif pos == movie.pos:
                hold = self.no_hold
            else:
                continue
            hold_left = left-hold.get_width()/2
            hold_bottom = bottom+height-hold.get_height()
            screen.blit(hold, (hold_left, hold_bottom), hold.get_rect())
            if pos == movie.pos:
                self.toggle_hold_boundaries = (hold_left, hold_bottom, hold_left+hold.get_width(), hold_bottom+hold.get_height())

    def update_on_light_table(self,x,y):
        for left, bottom, right, top, pos_dist in self.eye_boundaries:
            if y >= bottom and y <= top and x >= left and x <= right:
                self.on_light_table[pos_dist] = not self.on_light_table[pos_dist]
                return True

    def update_loop_mode(self,x,y):
        left, bottom, right, top = self.loop_boundaries
        if y >= bottom and y <= top and x >= left and x <= right:
            self.loop_mode = not self.loop_mode
            return True

    def update_hold(self,x,y):
        if len(movie.frames) <= 1:
            return
        left, bottom, right, top = self.toggle_hold_boundaries
        if y >= bottom and y <= top and x >= left and x <= right:
            toggle_frame_hold()
            return True

    def new_delete_tool(self): return isinstance(layout.tool, NewDeleteTool) 

    def on_mouse_down(self,x,y):
        self.prevx = None
        if self.new_delete_tool():
            if self.x2frame(x) == movie.pos:
                layout.tool.frame_func()
                restore_tool() # we don't want multiple clicks in a row to delete lots of frames etc
            return
        if self.update_on_light_table(x,y):
            return
        if self.update_loop_mode(x,y):
            return
        if self.update_hold(x,y):
            return
        self.prevx = x
    def on_mouse_up(self,x,y):
        self.on_mouse_move(x,y)
    def on_mouse_move(self,x,y):
        if self.prevx is None:
            return
        if self.new_delete_tool():
            return
        prev_pos = self.x2frame(self.prevx)
        curr_pos = self.x2frame(x)
        if prev_pos is None and curr_pos is None:
            self.prevx = x
            return
        if curr_pos is not None and prev_pos is not None:
            pos_dist = prev_pos - curr_pos
        else:
            pos_dist = -1 if x > self.prevx else 1
        self.prevx = x
        if pos_dist != 0:
            append_seek_frame_history_item_if_frame_is_dirty()
            if self.loop_mode:
                new_pos = (movie.pos + pos_dist) % len(movie.frames)
            else:
                new_pos = min(max(0, movie.pos + pos_dist), len(movie.frames)-1)
            movie.seek_frame(new_pos)

class LayersArea:
    def init(self):
        left, bottom, width, height = self.rect
        max_height = height / MAX_LAYERS
        max_width = IWIDTH * (max_height / IHEIGHT)
        self.width = min(max_width, width)
        image_height = int(self.width * IHEIGHT / IWIDTH)

        self.prevy = None
        self.color_images = {}
        icon_height = min(int(screen.get_width() * 0.15*0.14), image_height / 2)
        self.eye_open = scale_image(pg.image.load('eye_open.png'), height=icon_height)
        self.eye_shut = scale_image(pg.image.load('eye_shut.png'), height=icon_height)
        self.light_on = scale_image(pg.image.load('light_on.png'), height=icon_height)
        self.light_off = scale_image(pg.image.load('light_off.png'), height=icon_height)
        self.locked = scale_image(pg.image.load('locked.png'), height=icon_height)
        self.unlocked = scale_image(pg.image.load('unlocked.png'), height=icon_height)
        self.eye_boundaries = []
        self.lit_boundaries = []
        self.lock_boundaries = []
        self.thumbnail_height = 0
    
    def cached_image(self, layer_pos, layer):
        class CachedLayerThumbnail(CachedItem):
            def __init__(s, color=None):
                s.color = color
            def compute_key(s):
                frame = layer.frame(movie.pos) # note that we compute the thumbnail even if the layer is invisible
                return (frame.cache_id_version(),), ('single-layer-thumbnail', self.width, s.color)
            def compute_value(se):
                if se.color is None:
                    image = scale_image(movie._blit_layers([layer], movie.pos, include_invisible=True), self.width)
                    self.thumbnail_height = image.get_height()
                    return image
                image = cache.fetch(CachedLayerThumbnail()).copy()
                s = pg.Surface((image.get_width(), image.get_height()), pg.SRCALPHA)
                s.fill(BACKGROUND)
                if not self.color_images:
                    above_image = pg.Surface((image.get_width(), image.get_height()))
                    above_image.set_alpha(128)
                    above_image.fill(LAYERS_ABOVE)
                    below_image = pg.Surface((image.get_width(), image.get_height()))
                    below_image.set_alpha(128)
                    below_image.fill(LAYERS_BELOW)
                    self.color_images = {LAYERS_ABOVE: above_image, LAYERS_BELOW: below_image}
                image.blit(self.color_images[se.color], (0,0))
                image.set_alpha(128)
                s.blit(image, (0,0))
                return s

        if layer_pos > movie.layer_pos:
            color = LAYERS_ABOVE
        elif layer_pos < movie.layer_pos:
            color = LAYERS_BELOW
        else:
            color = None
        return cache.fetch(CachedLayerThumbnail(color))

    def draw(self):
        self.eye_boundaries = []
        self.lit_boundaries = []
        self.lock_boundaries = []

        left, bottom, width, height = self.rect
        top = bottom + width

        for layer_pos, layer in reversed(list(enumerate(movie.layers))):
            border = 1 + (layer_pos == movie.layer_pos)*2
            image = self.cached_image(layer_pos, layer)
            image_left = left + (width - image.get_width())/2
            screen.blit(image, (image_left, bottom), image.get_rect()) 
            pygame.draw.rect(screen, PEN, (image_left, bottom, image.get_width(), image.get_height()), border)

            max_border = 3
            if len(movie.frames) > 1 and layer.visible and layout.timeline_area().combined_light_table_mask():
                lit = self.light_on if layer.lit else self.light_off
                screen.blit(lit, (left + width - lit.get_width() - max_border, bottom))
                self.lit_boundaries.append((left + width - lit.get_width() - max_border, bottom, left+width, bottom+lit.get_height(), layer_pos))
               
            eye = self.eye_open if layer.visible else self.eye_shut
            screen.blit(eye, (left + width - eye.get_width() - max_border, bottom + image.get_height() - eye.get_height() - max_border))
            self.eye_boundaries.append((left + width - eye.get_width() - max_border, bottom + image.get_height() - eye.get_height() - max_border, left+width, bottom+image.get_height(), layer_pos))

            lock = self.locked if layer.locked else self.unlocked
            lock_start = bottom + self.thumbnail_height/2 - lock.get_height()/2
            screen.blit(lock, (left, lock_start))
            self.lock_boundaries.append((left, lock_start, left+lock.get_width(), lock_start+lock.get_height(), layer_pos))

            bottom += image.get_height()

    def new_delete_tool(self): return isinstance(layout.tool, NewDeleteTool)

    def y2frame(self, y):
        if not self.thumbnail_height:
            return None
        _, bottom, _, _ = self.rect
        return len(movie.layers) - ((y-bottom) // self.thumbnail_height) - 1

    def update_on_light_table(self,x,y):
        for left, bottom, right, top, layer_pos in self.lit_boundaries:
            if y >= bottom and y <= top and x >= left and x <= right:
                movie.layers[layer_pos].toggle_lit() # no undo for this - it's not a "model change" but a "view change"
                movie.clear_cache()
                return True

    def update_visible(self,x,y):
        for left, bottom, right, top, layer_pos in self.eye_boundaries:
            if y >= bottom and y <= top and x >= left and x <= right:
                layer = movie.layers[layer_pos]
                layer.toggle_visible()
                history.append_item(ToggleHistoryItem(layer.toggle_visible))
                movie.clear_cache()
                return True

    def update_locked(self,x,y):
        for left, bottom, right, top, layer_pos in self.lock_boundaries:
            if y >= bottom and y <= top and x >= left and x <= right:
                layer = movie.layers[layer_pos]
                layer.toggle_locked()
                history.append_item(ToggleHistoryItem(layer.toggle_locked))
                movie.clear_cache()
                return True

    def on_mouse_down(self,x,y):
        self.prevy = None
        if self.new_delete_tool():
            if self.y2frame(y) == movie.layer_pos:
                layout.tool.layer_func()
                restore_tool() # we don't want multiple clicks in a row to delete lots of layers
            return
        if self.update_on_light_table(x,y):
            return
        if self.update_visible(x,y):
            return
        if self.update_locked(x,y):
            return
        f = self.y2frame(y)
        if f == movie.layer_pos:
            self.prevy = y
    def on_mouse_up(self,x,y):
        self.on_mouse_move(x,y)
    def on_mouse_move(self,x,y):
        if self.prevy is None:
            return
        if self.new_delete_tool():
            return
        prev_pos = self.y2frame(self.prevy)
        curr_pos = self.y2frame(y)
        if curr_pos is None or curr_pos < 0 or curr_pos >= len(movie.layers):
            return
        self.prevy = y
        pos_dist = curr_pos - prev_pos
        if pos_dist != 0:
            append_seek_frame_history_item_if_frame_is_dirty()
            new_pos = min(max(0, movie.layer_pos + pos_dist), len(movie.layers)-1)
            movie.seek_layer(new_pos)

def get_last_modified(filenames):
    f2mtime = {}
    for f in filenames:
        s = os.stat(f)
        f2mtime[f] = s.st_mtime
    return list(sorted(f2mtime.keys(), key=lambda f: f2mtime[f]))[-1]

class MovieListArea:
    def __init__(self):
        self.show_pos = None
        self.prevx = None
        self.reload()
        self.histories = {}
    def delete_current_history(self):
        del self.histories[self.clips[self.clip_pos]]
    def reload(self):
        self.clips = []
        self.images = []
        single_image_height = screen.get_height() * MOVIES_Y_SHARE
        for clipdir in get_clip_dirs():
            fulldir = os.path.join(WD, clipdir)
            with open(os.path.join(fulldir, CLIP_FILE), 'r') as clipfile:
                clip = json.loads(clipfile.read())
            curr_frame = clip['frame_pos']
            frame_file = os.path.join(fulldir, FRAME_FMT % curr_frame)
            image = pg.image.load(frame_file) if os.path.exists(frame_file) else layout.drawing_area().new_frame()
            self.images.append(scale_image(image, height=single_image_height))
            self.clips.append(fulldir)
        self.clip_pos = 0 
    def draw(self):
        _, _, width, _ = self.rect
        left = 0
        first = True
        pos = self.show_pos if self.show_pos is not None else self.clip_pos
        for image in self.images[pos:]:
            border = 1 + first*2
            if first and pos == self.clip_pos:
                try:
                    image = movie.get_thumbnail(movie.pos, image.get_width(), image.get_height()) 
                    self.images[pos] = image # this keeps the image correct when scrolled out of clip_pos
                    # (we don't self.reload() upon scrolling so self.images can go stale when the current
                    # clip is modified)
                except:
                    pass
            first = False
            self.subsurface.blit(image, (left, 0), image.get_rect()) 
            pygame.draw.rect(self.subsurface, PEN, (left, 0, image.get_width(), image.get_height()), border)
            left += image.get_width()
            if left >= width:
                break
    def new_delete_tool(self): return isinstance(layout.tool, NewDeleteTool) 
    def x2frame(self, x):
        if not self.images or x is None:
            return None
        left, _, _, _ = self.rect
        return (x-left) // self.images[0].get_width()
    def on_mouse_down(self,x,y):
        if self.new_delete_tool():
            if self.x2frame(x) == 0:
                layout.tool.clip_func()
                restore_tool()
            return
        self.prevx = x
        self.show_pos = self.clip_pos
    def on_mouse_move(self,x,y):
        if self.prevx is None:
            self.prevx = x # this happens eg when a new_delete_tool is used upon mouse down
            # and then the original tool is restored
            self.show_pos = self.clip_pos
        if self.new_delete_tool():
            return
        prev_pos = self.x2frame(self.prevx)
        curr_pos = self.x2frame(x)
        if prev_pos is None and curr_pos is None:
            self.prevx = x
            return
        if curr_pos is not None and prev_pos is not None:
            pos_dist = prev_pos - curr_pos
        else:
            pos_dist = -1 if x > self.prevx else 1
        self.prevx = x
        self.show_pos = min(max(0, self.show_pos + pos_dist), len(self.clips)-1) 
    def on_mouse_up(self,x,y):
        self.on_mouse_move(x,y)
        # opening a movie is a slow operation so we don't want it to be "too interactive"
        # (like timeline scrolling) - we wait for the mouse-up event to actually open the clip
        self.open_clip(self.show_pos)
        self.prevx = None
        self.show_pos = None
    def open_clip(self, clip_pos):
        if clip_pos == self.clip_pos:
            return
        global movie
        movie.save_before_closing()
        movie = Movie(self.clips[clip_pos])
        self.clip_pos = clip_pos
        self.open_history(clip_pos)
    def open_history(self, clip_pos):
        global history
        history = self.histories.get(self.clips[clip_pos], History())
    def save_history(self):
        if self.clips:
            self.histories[self.clips[self.clip_pos]] = history

class ToolSelectionButton:
    def __init__(self, tool):
        self.tool = tool
    def draw(self):
        pg.draw.rect(screen, SELECTED if self.tool is layout.full_tool else UNDRAWABLE, self.rect)
        self.tool.tool.draw(self.rect,self.tool.cursor[1])
    def on_mouse_down(self,x,y):
        set_tool(self.tool)
    def on_mouse_up(self,x,y): pass
    def on_mouse_move(self,x,y): pass

class TogglePlaybackButton(Button):
    def __init__(self, play_icon, pause_icon):
        self.play = play_icon
        self.pause = pause_icon
        self.scaled = False
    def draw(self):
        left, bottom, width, height = self.rect
        if not self.scaled:
            def scale(image):
                scaled_width, scaled_height = scale_and_preserve_aspect_ratio(image.get_width(), image.get_height(), width, height)
                return scale_image(image, scaled_width, scaled_height)
            self.play = scale(self.play)
            self.pause = scale(self.pause)
            self.scaled = True
            
        icon = self.pause if layout.is_playing else self.play
        screen.blit(icon, (left + (width-icon.get_width())/2, bottom + height - icon.get_height()))
    def on_mouse_down(self,x,y):
        toggle_playing()
    def on_mouse_up(self,x,y): pass
    def on_mouse_move(self,x,y): pass

Tool = collections.namedtuple('Tool', ['tool', 'cursor', 'chars'])

class Frame:
    def __init__(self, dir, layer_id=None, frame_id=None):
        self.dir = dir
        self.layer_id = layer_id
        if frame_id is not None: # id - load the surfaces from the directory
            self.id = frame_id
            for surf_id in self.surf_ids():
                setattr(self,surf_id,None)
                for fname in self.filenames_png_bmp(surf_id):
                    if os.path.exists(fname):
                        setattr(self,surf_id,pygame.image.load(fname))
                        break
        else:
            self.id = str(uuid.uuid1())
            self.color = None
            self.lines = None

        # we don't aim to maintain a "perfect" dirty flag such as "doing 5 things and undoing
        # them should result in dirty==False." The goal is to avoid gratuitous saving when
        # scrolling thru the timeline, which slows things down and prevents reopening
        # clips at the last actually-edited frame after exiting the program
        self.dirty = False
        # similarly to dirty, version isn't a perfect version number; we're fine with it
        # going up instead of back down upon undo, or going up by more than 1 upon a single
        # editing operation. the version number is used for knowing when a cache hit
        # would produce stale data; if we occasionally evict valid data it's not as bad
        # as for hits to occasionally return stale data
        self.version = 0
        self.hold = False

        cache.update_id(self.cache_id(), self.version)

        self.compression_subprocess = None

    def __del__(self):
        cache.delete_id(self.cache_id())

    def empty(self): return self.color is None

    def _create_surfaces_if_needed(self):
        if not self.empty():
            return
        self.color = layout.drawing_area().new_frame()
        self.lines = pg.Surface((self.color.get_width(), self.color.get_height()), pygame.SRCALPHA)
        self.lines.fill(PEN)
        pygame.surfarray.pixels_alpha(self.lines)[:] = 0

    def get_content(self): return self.color.copy(), self.lines.copy()
    def set_content(self, content):
        color, lines = content
        self.color = color.copy()
        self.lines = lines.copy()
    def clear(self):
        self.color = None
        self.lines = None

    def increment_version(self):
        self._create_surfaces_if_needed()
        self.dirty = True
        self.version += 1
        cache.update_id(self.cache_id(), self.version)

    def surf_ids(self): return ['lines','color']
    def get_width(self): return empty_frame().color.get_width()
    def get_height(self): return empty_frame().color.get_height()
    def get_rect(self): return empty_frame().color.get_rect()

    def surf_by_id(self, surface_id):
        s = getattr(self, surface_id)
        return s if s is not None else empty_frame().surf_by_id(surface_id)

    def surface(self):
        if self.empty():
            return empty_frame().color
        s = self.color.copy()
        s.blit(self.lines, (0, 0), (0, 0, s.get_width(), s.get_height()))
        return s

    def filenames_png_bmp(self,surface_id):
        fname = f'{self.id}-{surface_id}.'
        if self.layer_id:
            fname = os.path.join(f'layer-{self.layer_id}', fname)
        fname = os.path.join(self.dir, fname)
        return fname+'png', fname+'bmp'
    def wait_for_compression_to_finish(self):
        if self.compression_subprocess:
            self.compression_subprocess.wait()
        self.compression_subprocess = None
    def save(self):
        if self.dirty:
            self.wait_for_compression_to_finish()
            fnames = []
            for surf_id in self.surf_ids():
                fname_png, fname_bmp = self.filenames_png_bmp(surf_id)
                pygame.image.save(self.surf_by_id(surf_id), fname_bmp)
                fnames += [fname_bmp, fname_png]
            self.compression_subprocess = subprocess.Popen([sys.executable, sys.argv[0], 'compress-and-remove']+fnames)
            self.dirty = False
    def delete(self):
        self.wait_for_compression_to_finish()
        for surf_id in self.surf_ids():
            for fname in self.filenames_png_bmp(surf_id):
                if os.path.exists(fname):
                    os.unlink(fname)

    def size(self):
        # a frame is 2 RGBA surfaces
        return (self.get_width() * self.get_height() * 8) if not self.empty() else 0

    def cache_id(self): return (self.id, self.layer_id) if not self.empty() else None
    def cache_id_version(self): return self.cache_id(), self.version

_empty_frame = Frame('')
def empty_frame():
    _empty_frame._create_surfaces_if_needed()
    return _empty_frame

class Layer:
    def __init__(self, frames, dir, layer_id=None):
        self.dir = dir
        self.frames = frames
        self.id = layer_id if layer_id else str(uuid.uuid1())
        self.lit = True
        self.visible = True
        self.locked = False
        for frame in frames:
            frame.layer_id = self.id
        subdir = self.subdir()
        if not os.path.isdir(subdir):
            os.makedirs(subdir)

    def surface_pos(self, pos):
        while self.frames[pos].hold:
            pos -= 1
        return pos

    def frame(self, pos): # return the closest frame in the past where hold is false
        return self.frames[self.surface_pos(pos)]

    def subdir(self): return os.path.join(self.dir, f'layer-{self.id}')
    def deleted_subdir(self): return self.subdir() + '-deleted'

    def delete(self):
        for frame in self.frames:
            frame.wait_for_compression_to_finish()
        os.rename(self.subdir(), self.deleted_subdir())
    def undelete(self): os.rename(self.deleted_subdir(), self.subdir())

    def toggle_locked(self): self.locked = not self.locked
    def toggle_lit(self): self.lit = not self.lit
    def toggle_visible(self):
        self.visible = not self.visible
        self.lit = self.visible

class Movie:
    def __init__(self, dir):
        self.dir = dir
        if not os.path.isdir(dir): # new clip
            os.makedirs(dir)
            self.frames = [Frame(self.dir)]
            self.pos = 0
            self.layers = [Layer(self.frames, dir)]
            self.layer_pos = 0
            self.frames[0].save()
            self.save_meta()
        else:
            with open(os.path.join(dir, CLIP_FILE), 'r') as clip_file:
                clip = json.loads(clip_file.read())
            frame_ids = clip['frame_order']
            layer_ids = clip['layer_order']
            holds = clip['hold']
            visible = clip.get('layer_visible', [True]*len(layer_ids))
            locked = clip.get('layer_locked', [False]*len(layer_ids))

            self.layers = []
            for layer_index, layer_id in enumerate(layer_ids):
                frames = []
                for frame_index, frame_id in enumerate(frame_ids):
                    frame = Frame(dir, layer_id, frame_id)
                    frame.hold = holds[layer_index][frame_index]
                    frames.append(frame)
                layer = Layer(frames, dir, layer_id)
                layer.visible = visible[layer_index]
                layer.locked = locked[layer_index]
                self.layers.append(layer)

            self.pos = clip['frame_pos']
            self.layer_pos = clip['layer_pos']
            self.frames = self.layers[self.layer_pos].frames

    def toggle_hold(self):
        pos = self.pos
        assert pos != 0 # in loop mode one might expect to be able to hold the last frame and have it shown
        # at the next frame, but that would create another surprise edge case - what happens when they're all held?..
        # better this milder surprise...
        if self.frames[pos].hold: # this frame's surface wasn't displayed - save the one that was
            self.frame(pos).save()
        else: # this frame was displayed and now won't be - save it before displaying the held one
            self.frames[pos].save()
        self.frames[pos].hold = not self.frames[pos].hold
        self.clear_cache()
        self.save_meta()

    def save_meta(self):
        # TODO: save light table settings
        clip = {
            'frame_pos':self.pos,
            'layer_pos':self.layer_pos,
            'frame_order':[frame.id for frame in self.frames],
            'layer_order':[layer.id for layer in self.layers],
            'layer_visible':[layer.visible for layer in self.layers],
            'layer_locked':[layer.locked for layer in self.layers],
            'hold':[[frame.hold for frame in layer.frames] for layer in self.layers],
        }
        text = json.dumps(clip,indent=2)
        with open(os.path.join(self.dir, CLIP_FILE), 'w') as clip_file:
            clip_file.write(text)

    def frame(self, pos):
        return self.layers[self.layer_pos].frame(pos)

    def get_mask(self, pos, rgb, transparency, key=False):
        # ignore invisible layers
        layers = [layer for layer in self.layers if layer.visible]
        # ignore the layers where the frame at the current position is an alias for the frame at the requested position
        # (it's visually noisy to see the same lines colored in different colors all over)
        def lines_lit(layer): return layer.lit and layer.surface_pos(self.pos) != layer.surface_pos(pos)

        class CachedMaskAlpha:
            def compute_key(_):
                frames = [layer.frame(pos) for layer in layers]
                lines = tuple([lines_lit(layer) for layer in layers])
                return tuple([frame.cache_id_version() for frame in frames if not frame.empty()]), ('mask-alpha', lines)
            def compute_value(_):
                alpha = np.zeros((empty_frame().get_width(), empty_frame().get_height()))
                for layer in layers:
                    frame = layer.frame(pos)
                    pen = pygame.surfarray.pixels_alpha(frame.surf_by_id('lines'))
                    color = pygame.surfarray.pixels_alpha(frame.surf_by_id('color'))
                    # hide the areas colored by this layer, and expose the lines of these layer (the latter, only if it's lit and not held)
                    alpha[:] = np.minimum(255-color, alpha)
                    if lines_lit(layer):
                        alpha[:] = np.maximum(pen, alpha)
                return alpha

        class CachedMask:
            def compute_key(_):
                id2version, computation = CachedMaskAlpha().compute_key()
                return id2version, ('mask', rgb, transparency, computation)
            def compute_value(_):
                mask_surface = pygame.Surface((empty_frame().get_width(), empty_frame().get_height()), pygame.SRCALPHA)
                pygame.surfarray.pixels3d(mask_surface)[:] = np.array(rgb)
                pg.surfarray.pixels_alpha(mask_surface)[:] = cache.fetch(CachedMaskAlpha())
                mask_surface.set_alpha(int(transparency*255))
                return mask_surface

        if key:
            return CachedMask().compute_key()
        return cache.fetch(CachedMask())

    def _visible_layers_id2version(self, layers, pos):
        frames = [layer.frame(pos) for layer in layers if layer.visible]
        return tuple([frame.cache_id_version() for frame in frames if not frame.empty()])

    def get_thumbnail(self, pos, width, height):

        class CachedThumbnail(CachedItem):
            def compute_key(_): return self._visible_layers_id2version(self.layers, pos), ('thumbnail', width, height)
            def compute_value(_):
                h = int(screen.get_height() * 0.15)
                w = int(h * IWIDTH / IHEIGHT)
                if w == width and h == height:
                    class CachedBottomThumbnail:
                        def compute_key(_): return self._visible_layers_id2version(self.layers[:self.layer_pos], pos), ('thumbnail', width, height)
                        def compute_value(_): return scale_image(self.curr_bottom_layers_surface(pos, highlight=False), width, height)
                    class CachedTopThumbnail: # top layers are transparent so it's a distinct cache category from normal "thumbnails"
                        def compute_key(_): return self._visible_layers_id2version(self.layers[self.layer_pos+1:], pos), ('transparent-thumbnail', width, height)
                        def compute_value(_): return scale_image(self.curr_top_layers_surface(pos, highlight=False), width, height)
                    class CachedMiddleThumbnail:
                        def compute_key(_): return self._visible_layers_id2version([self.layers[self.layer_pos]], pos), ('transparent-thumbnail', width, height)
                        def compute_value(_): return scale_image(self.frame(pos).surface(), width, height)

                    s = cache.fetch(CachedBottomThumbnail()).copy()
                    if self.layers[self.layer_pos].visible:
                        s.blit(cache.fetch(CachedMiddleThumbnail()), (0, 0))
                    s.blit(cache.fetch(CachedTopThumbnail()), (0, 0))
                    return s
                else:
                    return scale_image(self.get_thumbnail(pos, w, h), width, height)

        return cache.fetch(CachedThumbnail())

    def clear_cache(self):
        layout.drawing_area().fading_mask = None

    def seek_frame_and_layer(self,pos,layer_pos):
        assert pos >= 0 and pos < len(self.frames)
        assert layer_pos >= 0 and layer_pos < len(self.layers)
        if pos == self.pos and layer_pos == self.layer_pos:
            return
        self.frame(self.pos).save()
        self.pos = pos
        self.layer_pos = layer_pos
        self.frames = self.layers[layer_pos].frames
        self.clear_cache()
        self.save_meta()

    def seek_frame(self,pos): self.seek_frame_and_layer(pos, self.layer_pos)
    def seek_layer(self,layer_pos): self.seek_frame_and_layer(self.pos, layer_pos)

    def next_frame(self): self.seek_frame((self.pos + 1) % len(self.frames))
    def prev_frame(self): self.seek_frame((self.pos - 1) % len(self.frames))

    def next_layer(self): self.seek_layer((self.layer_pos + 1) % len(self.layers))
    def prev_layer(self): self.seek_layer((self.layer_pos - 1) % len(self.layers))

    def insert_frame(self):
        frame_id = str(uuid.uuid1())
        for layer in self.layers:
            frame = Frame(self.dir, layer.id)
            frame.id = frame_id
            frame.hold = layer is not self.layers[self.layer_pos] # by default, hold the other layers' frames
            layer.frames.insert(self.pos+1, frame)
        self.next_frame()

    def insert_layer(self):
        frames = [Frame(self.dir, None, frame.id) for frame in self.frames]
        layer = Layer(frames, self.dir)
        self.layers.insert(self.layer_pos+1, layer)
        self.next_layer()

    def reinsert_frame_at_pos(self, pos, removed_frame_data):
        assert pos >= 0 and pos <= len(self.frames)
        removed_frames, first_holds = removed_frame_data
        assert len(removed_frames) == len(self.layers)
        assert len(first_holds) == len(self.layers)

        self.frame(self.pos).save()
        self.pos = pos

        for layer, frame, hold in zip(self.layers, removed_frames, first_holds):
            layer.frames[0].hold = hold
            layer.frames.insert(self.pos, frame)
            frame.save()

        self.clear_cache()
        self.save_meta()

    def reinsert_layer_at_pos(self, layer_pos, removed_layer):
        assert layer_pos >= 0 and layer_pos <= len(self.layers)
        assert len(removed_layer.frames) == len(self.frames)

        self.frame(self.pos).save()
        self.layer_pos = layer_pos

        self.layers.insert(self.layer_pos, removed_layer)
        removed_layer.undelete()

        self.clear_cache()
        self.save_meta()

    def remove_frame(self, at_pos=-1, new_pos=-1):
        if len(self.frames) <= 1:
            return

        self.clear_cache()

        if at_pos == -1:
            at_pos = self.pos
        else:
            self.frame(self.pos).save()
        self.pos = at_pos

        removed_frames = []
        first_holds = []
        for layer in self.layers:
            removed = layer.frames[self.pos]
    
            del layer.frames[self.pos]
            removed.delete()
            removed.dirty = True # otherwise reinsert_frame_at_pos() calling frame.save() will not save the frame to disk,
            # which would be bad we just called frame.delete() to delete it from the disk

            removed_frames.append(removed)
            first_holds.append(layer.frames[0].hold)

            layer.frames[0].hold = False # could have been made true if we deleted frame 0
            # and frame 1 had hold==True - now this wouldn't make sense

        if self.pos >= len(self.frames):
            self.pos = len(self.frames)-1

        if new_pos >= 0:
            self.pos = new_pos

        self.save_meta()

        return removed_frames, first_holds

    def remove_layer(self, at_pos=-1, new_pos=-1):
        if len(self.layers) <= 1:
            return

        self.clear_cache()

        if at_pos == -1:
            at_pos = self.layer_pos
        else:
            self.frame(self.pos).save()
        self.layer_pos = at_pos

        removed = self.layers[self.layer_pos]
        del self.layers[self.layer_pos]
        removed.delete()

        if self.layer_pos >= len(self.layers):
            self.layer_pos = len(self.layers)-1

        if new_pos >= 0:
            self.layer_pos = new_pos

        self.save_meta()

        return removed

    def curr_frame(self):
        return self.frame(self.pos)

    def curr_layer(self):
        return self.layers[self.layer_pos]

    def edit_curr_frame(self):
        f = self.frame(self.pos)
        f.increment_version()
        return f

    def _blit_layers(self, layers, pos, transparent=False, include_invisible=False):
        f = self.curr_frame()
        if transparent:
            s = pg.Surface((f.get_width(), f.get_height()), pg.SRCALPHA)
        else:
            s = make_surface(f.get_width(), f.get_height())
            s.fill(BACKGROUND)
        surfaces = []
        for layer in layers:
            if not layer.visible and not include_invisible:
                continue
            f = layer.frame(pos)
            surfaces.append(f.surf_by_id('color'))
            surfaces.append(f.surf_by_id('lines'))
        s.blits([(surface, (0, 0), (0, 0, surface.get_width(), surface.get_height())) for surface in surfaces])
        return s

    def _set_undrawable_layers_grid(self, s):
        alpha = pg.surfarray.pixels3d(s)
        alpha[::WIDTH*3, ::WIDTH*3, :] = 0
        alpha[1::WIDTH*3, ::WIDTH*3, :] = 0
        alpha[:1:WIDTH*3, ::WIDTH*3, :] = 0
        alpha[1:1:WIDTH*3, ::WIDTH*3, :] = 0

    def curr_bottom_layers_surface(self, pos, highlight):
        class CachedBottomLayers:
            def compute_key(_):
                return self._visible_layers_id2version(self.layers[:self.layer_pos], pos), 'blit-bottom-layers' if not highlight else 'bottom-layers-highlighted'
            def compute_value(_):
                layers = self._blit_layers(self.layers[:self.layer_pos], pos, transparent=True)
                s = pg.Surface((layers.get_width(), layers.get_height()), pg.SRCALPHA)
                s.fill(BACKGROUND)
                scale = layout.drawing_area().scale
                if self.layer_pos == 0:
                    return scale(s)
                if not highlight:
                    s.blit(layers, (0, 0))
                    return scale(s)
                layers.set_alpha(128)
                below_image = pg.Surface((layers.get_width(), layers.get_height()), pg.SRCALPHA)
                below_image.set_alpha(128)
                below_image.fill(LAYERS_BELOW)
                alpha = pg.surfarray.array_alpha(layers)
                layers.blit(below_image, (0,0))
                pg.surfarray.pixels_alpha(layers)[:] = alpha
                self._set_undrawable_layers_grid(layers)
                s.blit(layers, (0,0))

                return scale(s)

        return cache.fetch(CachedBottomLayers())

    def curr_top_layers_surface(self, pos, highlight):
        class CachedTopLayers:
            def compute_key(_):
                return self._visible_layers_id2version(self.layers[self.layer_pos+1:], pos), 'blit-top-layers' if not highlight else 'top-layers-highlighted'
            def compute_value(_):
                layers = self._blit_layers(self.layers[self.layer_pos+1:], pos, transparent=True)
                scale = layout.drawing_area().scale
                if not highlight or self.layer_pos == len(self.layers)-1:
                    return scale(layers)
                layers.set_alpha(128)
                s = pg.Surface((layers.get_width(), layers.get_height()), pg.SRCALPHA)
                s.fill(BACKGROUND)
                above_image = pg.Surface((layers.get_width(), layers.get_height()), pg.SRCALPHA)
                above_image.set_alpha(128)
                above_image.fill(LAYERS_ABOVE)
                alpha = pg.surfarray.array_alpha(layers)
                layers.blit(above_image, (0,0))
                self._set_undrawable_layers_grid(layers)
                s.blit(layers, (0,0))
                pg.surfarray.pixels_alpha(s)[:] = alpha
                s.set_alpha(192)

                return scale(s)

        return cache.fetch(CachedTopLayers())

    def save_gif_and_pngs(self):
        with imageio.get_writer(self.dir + '.gif', fps=FRAME_RATE, loop=0) as writer:
            for i in range(len(self.frames)):
                frame = self._blit_layers(self.layers, i)
                pixels = np.transpose(pygame.surfarray.pixels3d(frame), [1,0,2])
                writer.append_data(pixels)
                imageio.imwrite(os.path.join(self.dir, FRAME_FMT%i), pixels) 

    def garbage_collect_layer_dirs(self):
        # we don't remove deleted layers from the disk when they're deleted since if there are a lot
        # of frames, this could be slow. those deleted layers not later un-deleted by the removal ops being undone
        # will be garbage-collected here
        for f in os.listdir(self.dir):
            full = os.path.join(self.dir, f)
            if f.endswith('-deleted') and f.startswith('layer-') and os.path.isdir(full):
                shutil.rmtree(full)

    def save_before_closing(self):
        layout.movie_list_area().save_history()
        global history
        history = History()
        self.frame(self.pos).save()
        self.save_meta()
        self.save_gif_and_pngs()
        self.garbage_collect_layer_dirs()
        for layer in self.layers:
            for frame in layer.frames:
                frame.wait_for_compression_to_finish()

class SeekFrameHistoryItem:
    def __init__(self, pos, layer_pos):
        self.pos = pos
        self.layer_pos = layer_pos
    def undo(self):
        redo = SeekFrameHistoryItem(movie.pos, movie.layer_pos)
        movie.seek_frame_and_layer(self.pos, self.layer_pos)
        return redo
    def __str__(self): return f'SeekFrameHistoryItem(restoring pos to {self.pos} and layer_pos to {self.layer_pos})'

class InsertFrameHistoryItem:
    def __init__(self, pos): self.pos = pos
    def undo(self):
        # normally remove_frame brings you to the next frame after the one you removed.
        # but when undoing insert_frame, we bring you to the previous frame after the one
        # you removed - it's the one where you inserted the frame we're now removing to undo
        # the insert, so this is where we should go to bring you back in time.
        removed_frame_data = movie.remove_frame(at_pos=self.pos, new_pos=max(0, self.pos-1))
        return RemoveFrameHistoryItem(self.pos, removed_frame_data)
    def __str__(self):
        return f'InsertFrameHistoryItem(removing at pos {self.pos})'

class RemoveFrameHistoryItem:
    def __init__(self, pos, removed_frame_data):
        self.pos = pos
        self.removed_frame_data = removed_frame_data
    def undo(self):
        movie.reinsert_frame_at_pos(self.pos, self.removed_frame_data)
        return InsertFrameHistoryItem(self.pos)
    def __str__(self):
        return f'RemoveFrameHistoryItem(inserting at pos {self.pos})'
    def byte_size(self):
        frames, holds = self.removed_frame_data
        return sum([f.size() for f in frames])

class InsertLayerHistoryItem:
    def __init__(self, layer_pos): self.layer_pos = layer_pos
    def undo(self):
        removed_layer = movie.remove_layer(at_pos=self.layer_pos, new_pos=max(0, self.layer_pos-1))
        return RemoveLayerHistoryItem(self.layer_pos, removed_layer)
    def __str__(self):
        return f'InsertLayerHistoryItem(removing layer {self.layer_pos})'

class RemoveLayerHistoryItem:
    def __init__(self, layer_pos, removed_layer):
        self.layer_pos = layer_pos
        self.removed_layer = removed_layer
    def undo(self):
        movie.reinsert_layer_at_pos(self.layer_pos, self.removed_layer)
        return InsertLayerHistoryItem(self.layer_pos)
    def __str__(self):
        return f'RemoveLayerHistoryItem(inserting layer {self.layer_pos})'
    def byte_size(self):
        return sum([f.size() for f in self.removed_layer.frames])

class ToggleHoldHistoryItem:
    def __init__(self, pos, layer_pos):
        self.pos = pos
        self.layer_pos = layer_pos
    def undo(self):
        if movie.pos != self.pos or movie.layer_pos != self.layer_pos:
            print('WARNING: wrong pos for a toggle-hold history item - expected {self.pos} layer {self.layer_pos}, got {movie.pos} layer {movie.layer_pos}')
            movie.seek_frame_and_layer(self.pos, self.layer_pos)
        movie.toggle_hold()
        return self
    def __str__(self):
        return f'ToggleHoldHistoryItem(toggling hold at frame {self.pos} layer {self.layer_pos})'

class ToggleHistoryItem:
    def __init__(self, toggle_func): self.toggle_func = toggle_func
    def undo(self):
        self.toggle_func()
        return self
    def __str__(self):
        return f'ToggleHistoryItem({self.toggle_func.__qualname__})'

def append_seek_frame_history_item_if_frame_is_dirty():
    if history.undo:
        last_op = history.undo[-1]
        if not isinstance(last_op, SeekFrameHistoryItem):
            history.append_item(SeekFrameHistoryItem(movie.pos, movie.layer_pos))

def insert_frame():
    movie.insert_frame()
    history.append_item(InsertFrameHistoryItem(movie.pos))

def insert_layer():
    movie.insert_layer()
    history.append_item(InsertLayerHistoryItem(movie.layer_pos))

def remove_frame():
    if len(movie.frames) == 1:
        return
    pos = movie.pos
    removed_frame_data = movie.remove_frame()
    history.append_item(RemoveFrameHistoryItem(pos, removed_frame_data))

def remove_layer():
    if len(movie.layers) == 1:
        return
    layer_pos = movie.layer_pos
    removed_layer = movie.remove_layer()
    history.append_item(RemoveLayerHistoryItem(layer_pos, removed_layer))

def next_frame():
    if movie.pos >= len(movie.frames)-1 and not layout.timeline_area().loop_mode:
        return
    append_seek_frame_history_item_if_frame_is_dirty()
    movie.next_frame()

def prev_frame():
    if movie.pos <= 0 and not layout.timeline_area().loop_mode:
        return
    append_seek_frame_history_item_if_frame_is_dirty()
    movie.prev_frame()

def insert_clip():
    global movie
    movie.save_before_closing()
    movie = Movie(new_movie_clip_dir())
    movie.save_gif_and_pngs() # write out FRAME_FMT % 0 for MovieListArea.reload...
    layout.movie_list_area().reload()

def remove_clip():
    movie_list_area = layout.movie_list_area()
    if len(movie_list_area.clips) <= 1:
        return # we don't remove the last clip - if we did we'd need to create a blank one,
        # which is a bit confusing. [we can't remove the last frame in a timeline, either]
    global movie
    movie.save_before_closing()
    os.rename(movie.dir, movie.dir + '-deleted')
    movie_list_area.delete_current_history()
    movie_list_area.reload()

    new_clip_pos = 0
    movie = Movie(movie_list_area.clips[new_clip_pos])
    movie_list_area.open_history(new_clip_pos)

def toggle_playing(): layout.toggle_playing()

def toggle_loop_mode():
    timeline_area = layout.timeline_area()
    timeline_area.loop_mode = not timeline_area.loop_mode

def toggle_frame_hold():
    if movie.pos != 0 and not curr_layer_locked():
        movie.toggle_hold()
        history.append_item(ToggleHoldHistoryItem(movie.pos, movie.layer_pos))

def toggle_layer_lock():
    layer = movie.curr_layer()
    layer.toggle_locked()
    history.append_item(ToggleHistoryItem(layer.toggle_locked))

TOOLS = {
    'pencil': Tool(PenTool(), pencil_cursor, 'bB'),
    'eraser': Tool(PenTool(BACKGROUND, WIDTH), eraser_cursor, 'eE'),
    'eraser-medium': Tool(PenTool(BACKGROUND, MEDIUM_ERASER_WIDTH), eraser_medium_cursor, 'rR'),
    'eraser-big': Tool(PenTool(BACKGROUND, BIG_ERASER_WIDTH), eraser_big_cursor, 'tT'),
    'flashlight': Tool(FlashlightTool(), flashlight_cursor, 'fF'),
    # insert/remove frame are both a "tool" (with a special cursor) and a "function."
    # meaning, when it's used thru a keyboard shortcut, a frame is inserted/removed
    # without any more ceremony. but when it's used thru a button, the user needs to
    # actually press on the current image in the timeline to remove/insert. this,
    # to avoid accidental removes/inserts thru misclicks and a resulting confusion
    # (a changing cursor is more obviously "I clicked a wrong button, I should click
    # a different one" than inserting/removing a frame where you need to undo but to
    # do that, you need to understand what just happened)
    'insert-frame': Tool(NewDeleteTool(insert_frame, insert_clip, insert_layer), blank_page_cursor, ''),
    'remove-frame': Tool(NewDeleteTool(remove_frame, remove_clip, remove_layer), garbage_bin_cursor, ''),
}

FUNCTIONS = {
    'insert-frame': (insert_frame, '=+', pg.image.load('sheets.png')),
    'remove-frame': (remove_frame, '-_', pg.image.load('garbage.png')),
    'next-frame': (next_frame, '.<', None),
    'prev-frame': (prev_frame, ',>', None),
    'toggle-playing': (toggle_playing, '\r', None),
    'toggle-loop-mode': (toggle_loop_mode, 'c', None),
    'toggle-frame-hold': (toggle_frame_hold, 'h', None),
    'toggle-layer-lock': (toggle_layer_lock, 'l', None),
}

prev_tool = None
def set_tool(tool):
    global prev_tool
    prev = layout.full_tool
    layout.tool = tool.tool
    layout.full_tool = tool
    if not isinstance(prev.tool, NewDeleteTool):
        prev_tool = prev
    if tool.cursor:
        try_set_cursor(tool.cursor[0])

def restore_tool():
    set_tool(prev_tool)

def color_image(s, rgba):
    sc = s.copy()
    pixels = pg.surfarray.pixels3d(sc)
    for ch in range(3):
        pixels[:,:,ch] = (pixels[:,:,ch].astype(int)*rgba[ch])//255
    if rgba[-1] == 0:
        alphas = pg.surfarray.pixels_alpha(sc)
        alphas[:] = np.minimum(alphas[:], 255 - pixels[:,:,0])
    return sc

class Palette:
    def __init__(self, filename, rows=12, columns=3):
        s = pg.image.load(filename)
        color_hist = {}
        first_color_hit = {}
        white = (255,255,255)
        for y in range(s.get_height()):
            for x in range(s.get_width()):
                r,g,b,a = s.get_at((x,y))
                color = r,g,b
                if color not in first_color_hit:
                    first_color_hit[color] = (y / (s.get_height()/3))*s.get_width() + x
                if color != white:
                    color_hist[color] = color_hist.get(color,0) + 1

        colors = [[None for col in range(columns)] for row in range(rows)]
        colors[0] = [BACKGROUND+(0,), white+(255,), white+(255,)]
        color2popularity = dict(list(reversed(sorted(list(color_hist.items()), key=lambda x: x[1])))[:(rows-1)*columns])
        hit2color = [(first_hit, color) for color, first_hit in sorted(list(first_color_hit.items()), key=lambda x: x[1])]

        row = 1
        col = 0
        for hit, color in hit2color:
            if color in color2popularity:
                colors[row][col] = color + (255,)
                row+=1
                if row == rows:
                    row = 1
                    col += 1

        self.rows = rows
        self.columns = columns
        self.colors = colors

        self.init_cursors()

    def init_cursors(self):
        s = paint_bucket_cursor[0]
        self.cursors = [[None for col in range(self.columns)] for row in range(self.rows)]
        for row in range(self.rows):
            for col in range(self.columns):
                sc = color_image(s, self.colors[row][col])
                self.cursors[row][col] = (pg.cursors.Cursor((0,sc.get_height()-1), sc), color_image(paint_bucket_cursor[1], self.colors[row][col]))
                if self.colors[row][col][-1] == 0: # water tool
                    self.cursors[row][col] = (self.cursors[row][col][0], scale_image(pg.image.load('water-tool.png'), self.cursors[row][col][1].get_width()))


palette = Palette('palette.png')

def get_clip_dirs():
    '''returns the clip directories sorted by last modification time (latest first)'''
    wdfiles = os.listdir(WD)
    clipdirs = {}
    for d in wdfiles:
        try:
            if d.endswith('-deleted'):
                continue
            frame_order_file = os.path.join(os.path.join(WD, d), CLIP_FILE)
            s = os.stat(frame_order_file)
            clipdirs[d] = s.st_mtime
        except:
            continue

    return list(reversed(sorted(clipdirs.keys(), key=lambda d: clipdirs[d])))

TIMELINE_Y_SHARE = 0.15
LAYERS_Y_SHARE = 1-TIMELINE_Y_SHARE
MAX_LAYERS = 8
LAYERS_X_SHARE = LAYERS_Y_SHARE / MAX_LAYERS
TOOLBAR_X_SHARE = 0.15
DRAWING_AREA_X_SHARE = 1 - TOOLBAR_X_SHARE - LAYERS_X_SHARE
DRAWING_AREA_Y_SHARE = DRAWING_AREA_X_SHARE # preserve screen aspect ratio
MOVIES_X_SHARE = 1-TOOLBAR_X_SHARE-LAYERS_X_SHARE
MOVIES_Y_SHARE = 1-DRAWING_AREA_Y_SHARE-TIMELINE_Y_SHARE
def init_layout_basic():
    screen.fill(UNDRAWABLE)

    global layout
    layout = Layout()
    layout.add((TOOLBAR_X_SHARE, TIMELINE_Y_SHARE, DRAWING_AREA_X_SHARE, DRAWING_AREA_Y_SHARE), DrawingArea())

def init_layout_rest():
    layout.add((0, 0, 1, TIMELINE_Y_SHARE), TimelineArea())
    layout.add((TOOLBAR_X_SHARE, TIMELINE_Y_SHARE+DRAWING_AREA_Y_SHARE, MOVIES_X_SHARE, MOVIES_Y_SHARE), MovieListArea())
    layout.add((TOOLBAR_X_SHARE+DRAWING_AREA_X_SHARE, TIMELINE_Y_SHARE, LAYERS_X_SHARE, LAYERS_Y_SHARE), LayersArea())

    tools_width_height = [
        ('pencil', 0.33, 1),
        ('eraser-big', 0.27, 1),
        ('eraser-medium', 0.21, 0.8),
        ('eraser', 0.15, 0.6),
    ]
    offset = 0
    for tool, width, height in tools_width_height:
        layout.add((offset*0.15,0.85+(0.15*(1-height)),width*0.15, 0.15*height), ToolSelectionButton(TOOLS[tool]))
        offset += width
    color_w = 0.025*2
    i = 0

    layout.add((color_w*2, 0.85-color_w, color_w, color_w*1.5), ToolSelectionButton(TOOLS['flashlight']))
    
    for row,y in enumerate(np.arange(0.25,0.85-0.001,color_w)):
        for col,x in enumerate(np.arange(0,0.15-0.001,color_w)):            
            if row == len(palette.colors)-1 and col == 2:
                continue
            tool = Tool(PaintBucketTool(palette.colors[len(palette.colors)-row-1][col]), palette.cursors[len(palette.colors)-row-1][col], '')
            layout.add((x,y,color_w,color_w), ToolSelectionButton(tool))
            i += 1

    funcs_width = [
        ('insert-frame', 0.33),
        ('remove-frame', 0.33),
        ('play', 0.33)
    ]
    offset = 0
    for func, width in funcs_width:
        if func == 'play':
            button = TogglePlaybackButton(pg.image.load('play.png'), pg.image.load('pause.png'))
        else:
            button = ToolSelectionButton(TOOLS[func])
        layout.add((offset*0.15,0.15,width*0.15, 0.1), button)
        offset += width

    layout.draw()

def new_movie_clip_dir():
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return os.path.join(WD, now)

def default_clip_dir():
    clip_dirs = get_clip_dirs() 
    if not clip_dirs:
        # first clip - create a new directory
        return new_movie_clip_dir()
    else:
        return os.path.join(WD, clip_dirs[0])

init_layout_basic()
movie = Movie(default_clip_dir())
init_layout_rest()

# The history is "global" for all operations within a movie. In some (rare) animation programs
# there's a history per frame. One problem with this is how to undo timeline
# operations like frame deletions (do you have a separate undo function for this?)
# It's also somewhat less intuitive in that you might have long forgotten
# what you've done on some frame when you visit it and press undo one time
# too many
#
def byte_size(history_item):
    return getattr(history_item, 'byte_size', lambda: 128)()

def nop(history_item):
    return getattr(history_item, 'nop', lambda: False)()

class History:
    # a history is kept per movie. the size of the history is global - we don't
    # want to exceed a certain memory threshold for the history
    byte_size = 0
    
    def __init__(self):
        self.undo = []
        self.redo = []
        layout.drawing_area().fading_mask = None
        self.suggestions = None

    def __del__(self):
        for op in self.undo + self.redo:
            History.byte_size -= byte_size(op)

    def _merge_prev_suggestions(self):
        if self.suggestions: # merge them into one
            s = self.suggestions
            self.suggestions = None
            self.append_item(HistoryItemSet(list(reversed(s))))

    def append_suggestions(self, items):
        '''"suggestions" are multiple items taking us from a new state B to the old state A,
        for 2 suggestions - thru a single intermediate state S: B -> S -> A.

        there's a single opportunity to "accept" a suggestion by pressing 'undo' right after
        the suggestions were "made" by a call to append_suggestions(). in this case the history
        will have an item for B -> S and another one for S -> A. otherwise, the suggestions
        will be "merged" into a single B -> A HistoryItemSet (when new items or suggestions 
        are appended.)'''
        self._merge_prev_suggestions()
        if len(items) == 1:
            self.append_item(items[0])
        else:
            self.suggestions = items

    def append_item(self, item):
        if nop(item):
            return

        self._merge_prev_suggestions()

        self.undo.append(item)
        History.byte_size += byte_size(item) - sum([byte_size(op) for op in self.redo])
        self.redo = [] # forget the redo stack
        while self.undo and History.byte_size > MAX_HISTORY_BYTE_SIZE:
            History.byte_size -= byte_size(self.undo[0])
            del self.undo[0]

        layout.drawing_area().fading_mask = None # new operations invalidate old skeletons

    def undo_item(self):
        if self.suggestions:
            s = self.suggestions
            self.suggestions = None
            for item in s:
                self.append_item(item)

        if self.undo:
            last_op = self.undo[-1]
            redo = last_op.undo()
            History.byte_size += byte_size(redo) - byte_size(last_op)
            self.redo.append(redo)
            self.undo.pop()

        layout.drawing_area().fading_mask = None # changing canvas state invalidates old skeletons

    def redo_item(self):
        if self.redo:
            last_op = self.redo[-1]
            undo = last_op.undo()
            History.byte_size += byte_size(undo) - byte_size(last_op)
            self.undo.append(undo)
            self.redo.pop()

    def clear(self):
        History.byte_size -= sum([byte_size(op) for op in self.undo+self.redo])
        self.undo = []
        self.redo = []
        self.suggestions = None

def clear_history():
    history.clear()
    drawing_area = layout.drawing_area()
    fading_mask = drawing_area.new_frame()
    text_surface = font.render("Current Clip's\nUndo/Redo History\nDeleted!", True, (255, 0, 0), (255, 255, 255))
    fading_mask.blit(text_surface, ((fading_mask.get_width()-text_surface.get_width())/2, (fading_mask.get_height()-text_surface.get_height())/2))
    fading_mask.set_alpha(255)
    drawing_area.set_fading_mask(fading_mask)
    drawing_area.fade_per_frame = 255/(FADING_RATE*10)

history = History()

escape = False

PLAYBACK_TIMER_EVENT = pygame.USEREVENT + 1
SAVING_TIMER_EVENT = pygame.USEREVENT + 2
FADING_TIMER_EVENT = pygame.USEREVENT + 3

pygame.time.set_timer(PLAYBACK_TIMER_EVENT, 1000//FRAME_RATE) # we play back at 12 fps
pygame.time.set_timer(SAVING_TIMER_EVENT, 15*1000) # we save a copy of the current clip every 15 seconds
pygame.time.set_timer(FADING_TIMER_EVENT, 1000//FADING_RATE) # we save a copy of the current clip every 15 seconds

timer_events = [
    PLAYBACK_TIMER_EVENT,
    SAVING_TIMER_EVENT,
    FADING_TIMER_EVENT,
]

interesting_events = [
    pygame.KEYDOWN,
    pygame.MOUSEMOTION,
    pygame.MOUSEBUTTONDOWN,
    pygame.MOUSEBUTTONUP,
] + timer_events

keyboard_shortcuts_enabled = False # enabled by Ctrl-A; disabled by default to avoid "surprises"
# upon random banging on the keyboard

cut_frame_content = None

def copy_frame():
    global cut_frame_content
    cut_frame_content = movie.curr_frame().get_content()

def cut_frame():
    history_item = HistoryItemSet([HistoryItem('color'), HistoryItem('lines')])

    global cut_frame_content
    frame = movie.edit_curr_frame()
    cut_frame_content = frame.get_content()
    frame.clear()

    history_item.optimize()
    history.append_item(history_item)

def paste_frame():
    if not cut_frame_content:
        return

    history_item = HistoryItemSet([HistoryItem('color'), HistoryItem('lines')])

    movie.edit_curr_frame().set_content(cut_frame_content)

    history_item.optimize()
    history.append_item(history_item)

def process_keydown_event(event):
    ctrl = pg.key.get_mods() & pg.KMOD_CTRL
    shift = pg.key.get_mods() & pg.KMOD_SHIFT
    # Like Escape, Undo/Redo and Delete History are always available thru the keyboard [and have no other way to access them]
    if event.key == pg.K_SPACE:
        if ctrl:
            history.redo_item()
        else:
            history.undo_item()

    # Ctrl+Shift+Delete
    if event.key == pg.K_DELETE and ctrl and shift:
        clear_history()

    # Ctrl-C/X/V
    if ctrl:
        if event.key == pg.K_c:
            copy_frame()
        elif event.key == pg.K_x:
            cut_frame()
        elif event.key == pg.K_v:
            paste_frame()

    # other keyboard shortcuts are enabled/disabled by Ctrl-A
    global keyboard_shortcuts_enabled

    if keyboard_shortcuts_enabled:
        for tool in TOOLS.values():
            if event.key in [ord(c) for c in tool.chars]:
                set_tool(tool)

        for func, chars, _ in FUNCTIONS.values():
            if event.key in [ord(c) for c in chars]:
                func()
                
    if event.key == pygame.K_a and ctrl:
        keyboard_shortcuts_enabled = not keyboard_shortcuts_enabled
        print('Ctrl-A pressed -','enabling' if keyboard_shortcuts_enabled else 'disabling','keyboard shortcuts')

set_tool(TOOLS['pencil'])

layout.draw()
pygame.display.flip()

font = pygame.font.Font(size=screen.get_height()//10)

while not escape: 
 try:
  for event in pygame.event.get():
   #print(pg.event.event_name(event.type),tdiff(),event.type,pygame.key.get_mods())

   if event.type not in interesting_events:
       continue
   try:
      if event.type == pygame.KEYDOWN:

        if event.key == pygame.K_ESCAPE: # ESC pressed
            escape = True
            break

        if layout.is_pressed:
            continue # ignore keystrokes (except ESC) when a mouse tool is being used

        process_keydown_event(event)
        
      else:
          layout.on_event(event)

      # TODO: might be good to optimize repainting beyond "just repaint everything
      # upon every event"
      if layout.is_playing or (layout.drawing_area().fading_mask and event.type == FADING_TIMER_EVENT) or event.type not in timer_events:
        # don't repaint upon depressed mouse movement. this is important to avoid the pen
        # lagging upon "first contact" when a mouse motion event is sent before a mouse down
        # event at the same coordinate; repainting upon that mouse motion event loses time
        # when we should have been receiving the next x,y coordinates
        if event.type != pygame.MOUSEMOTION or layout.is_pressed:
            layout.draw()
            if not layout.is_playing:
                cache.collect_garbage()
        pygame.display.flip()
   except KeyboardInterrupt:
    print('Ctrl-C - exiting')
    escape = True
    break
   except:
    print('INTERNAL ERROR (printing and continuing)')
    import traceback
    traceback.print_exc()
 except KeyboardInterrupt:
  print('Ctrl-C - exiting')
  break
      
movie.save_before_closing()

pygame.display.quit()
pygame.quit()
