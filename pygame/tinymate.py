import pygame
import pygame.gfxdraw
import imageio
import winpath
import collections
import uuid
import math
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')  # turn off interactive backend
import io
import sys
import os
import json
import shutil
from scipy.interpolate import splprep, splev

# this requires numpy to be installed in addition to scikit-image
from skimage.morphology import flood_fill, binary_dilation, skeletonize
from scipy.ndimage import grey_dilation, grey_erosion, grey_opening, grey_closing
pg = pygame

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Tinymate")
#screen = pygame.display.set_mode((500, 500))

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
CURSOR_SIZE = int(screen.get_width() * 0.07)
MAX_HISTORY_BYTE_SIZE = 2*1024**3
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
    if len(intersections) > 0:
        len_first = intersections[0]
        len_last = len(ix) - intersections[-1]
        # look for clear alpha pixels along the path before the first and the last intersection - if we find some, we have >= 2 intersections
        two_or_more_intersections = len(np.where(line_alphas[intersections[0]:intersections[-1]] == 0)[0]) > 1

        first_short = two_or_more_intersections or len_first < len_last
        last_short = two_or_more_intersections or len_last <= len_first

        step=(u[-1]-u[0])/curve_length

        new_points = splev(np.arange(step*(intersections[0]+1) if first_short else u[0], step*(intersections[-1]) if last_short else u[-1], step), tck)
        return [new_points] + results

    # check if we'd like to attempt to close the line
    bbox_length = (np.max(x)-np.min(x))*2 + (np.max(y)-np.min(y))*2
    endpoints_dist = dist(0, -1)

    make_closed = len(points)>2 and should_make_closed(curve_length, bbox_length, endpoints_dist)

    if make_closed:
        orig_len = len(x)
        def half(ls):
            ls = list(ls)
            return ls[:-len(ls)//2]
        cx = np.array(list(x)+half([xi+0.001 for xi in x]))
        cy = np.array(list(y)+half([yi+0.001 for yi in y]))

        ctck, cu = splprep([cx, cy], s=len(cx)/5)
        add_result(ctck, cu[orig_len//2-1], cu[-1])
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

def scale_image(surface, width, height=None):
    if not height:
        height = int(surface.get_height() * width / surface.get_width())
    return pg.transform.smoothscale(surface, (width, height))

def minmax(v, minv, maxv):
    return min(maxv,max(minv,v))

def load_cursor(file, flip=False, size=CURSOR_SIZE, hot_spot=(0,1), min_alpha=192):
  surface = pg.image.load(file)
  surface = scale_image(surface, size, size*surface.get_height()/surface.get_width())#pg.transform.scale(surface, (CURSOR_SIZE, CURSOR_SIZE))
  if flip:
      surface = pg.transform.flip(surface, True, True)
  non_transparent_surface = surface.copy()
  for y in range(surface.get_height()):
      for x in range(surface.get_width()):
          r,g,b,a = surface.get_at((x,y))
          surface.set_at((x,y), (r,g,b,min(a,min_alpha)))
  hotx = minmax(int(hot_spot[0] * surface.get_width()), 0, surface.get_width()-1)
  hoty = minmax(int(hot_spot[1] * surface.get_height()), 0, surface.get_height()-1)
  return pg.cursors.Cursor((hotx, hoty), surface), non_transparent_surface

pencil_cursor = load_cursor('pencil.png')
pencil_cursor = (pencil_cursor[0], pg.image.load('pencil-tool.png'))
eraser_cursor = load_cursor('eraser.png')
eraser_cursor = (eraser_cursor[0], pg.image.load('eraser-tool.png'))
eraser_medium_cursor = load_cursor('eraser.png', size=int(CURSOR_SIZE*1.5))
eraser_medium_cursor = (eraser_medium_cursor[0], eraser_cursor[1])
eraser_big_cursor = load_cursor('eraser.png', size=int(CURSOR_SIZE*2))
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
        self.items = items
    def nop(self):
        for item in self.items:
            if not item.nop():
                return False
        return True
    def undo(self):
        return HistoryItemSet([item.undo() for item in self.items])
    def optimize(self):
        for item in self.items:
            item.optimize()
        self.items = [item for item in self.items if not item.nop()]
    def byte_size(self):
        return sum([item.byte_size() for item in self.items])

class Button:
    def __init__(self):
        self.button_surface = None
    def draw(self, rect, cursor_surface):
        left, bottom, width, height = rect
        _, _, w, h = cursor_surface.get_rect()
        scaled_width = w*height/h
        if not self.button_surface:
            surface = scale_image(cursor_surface, scaled_width, height)
            self.button_surface = surface
        screen.blit(self.button_surface, (left+width/2-scaled_width/2, bottom), (0, 0, scaled_width, height))

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

    def on_mouse_down(self, x, y):
        self.points = []
        self.bucket_color = None
        self.lines_array = pg.surfarray.pixels_alpha(movie.edit_curr_frame().surf_by_id('lines'))
        self.on_mouse_move(x,y)

    def on_mouse_up(self, x, y):
        self.lines_array = None
        drawing_area = layout.drawing_area()
        self.points.append((x-drawing_area.xmargin,y-drawing_area.ymargin))
        self.prev_drawn = None
        frame = movie.edit_curr_frame().surf_by_id('lines')
        lines = pygame.surfarray.pixels_alpha(frame)

        prev_history_item = None
        line_options = drawLines(frame.get_width(), frame.get_height(), self.points, self.width, suggest_options=not self.eraser, existing_lines=lines)
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
                flood_fill_color_based_on_lines(color_rgb, color_alpha, lines, x, y, self.bucket_color if self.bucket_color else BACKGROUND+(255,))
                history_item = HistoryItemSet([history_item, color_history_item])

            history_item.optimize()
            history.append_item(history_item)
            if not prev_history_item:
                prev_history_item = history_item
        
        if len(line_options)>1:
            if self.suggestion_mask is None:
                left, bottom, width, height = drawing_area.rect
                self.suggestion_mask = pg.Surface((width-drawing_area.xmargin*2, height-drawing_area.ymargin*2), pg.SRCALPHA)
                self.suggestion_mask.fill((0,255,0))
            alt_option = line_options[-2]
            pg.surfarray.pixels_alpha(self.suggestion_mask)[:] = 255-alt_option
            self.suggestion_mask.set_alpha(10)
            drawing_area.fading_mask = self.suggestion_mask
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
       if self.eraser and self.bucket_color is None and self.lines_array[x,y] != 255:
           self.bucket_color = movie.edit_curr_frame().surf_by_id('color').get_at((x,y))
       drawing_area = layout.drawing_area()
       draw_into = drawing_area.subsurface
       self.points.append((x-drawing_area.xmargin,y-drawing_area.ymargin))
       color = self.color if not self.eraser else (self.bucket_color if self.bucket_color else BACKGROUND)
       if self.prev_drawn:
            drawLine(draw_into, self.prev_drawn, (x,y), color, self.width)
       drawCircle(draw_into, x, y, color, self.circle_width)
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
        color_surface = movie.edit_curr_frame().surf_by_id('color')
        color_rgb = pg.surfarray.pixels3d(color_surface)
        color_alpha = pg.surfarray.pixels_alpha(color_surface)
        lines = pygame.surfarray.pixels_alpha(movie.edit_curr_frame().surf_by_id('lines'))
        
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
    dist = np.sqrt((xg - x)**2 + (yg - y)**2)
    d, maxdist = skeleton_to_distances(skeleton, x, y)
    d = (d == NO_PATH_DIST)*maxdist + (d != NO_PATH_DIST)*d # replace NO_PATH_DIST with maxdist
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
        color = pygame.surfarray.pixels3d(movie.curr_frame().surf_by_id('color'))
        lines = pygame.surfarray.pixels_alpha(movie.curr_frame().surf_by_id('lines'))
        fading_mask = skeletonize_color_based_on_lines(color, lines, x, y)
        if not fading_mask:
            return
        fading_mask.set_alpha(255)
        layout.drawing_area().fading_mask = fading_mask
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
            
def pen2mask(lines_list, rgb, transparency):
    mask_surface = pygame.Surface((empty_frame().get_width(), empty_frame().get_height()), pygame.SRCALPHA)
    mask = pygame.surfarray.pixels3d(mask_surface)
    pen = [pygame.surfarray.pixels_alpha(lines) for lines in lines_list]
    for ch in range(3):
        mask[:,:,ch] = rgb[ch]
    pygame.surfarray.pixels_alpha(mask_surface)[:] = np.maximum.reduce(pen) if pen else 0
    mask_surface.set_alpha(int(transparency*255))
    return mask_surface

class DrawingArea:
    def __init__(self):
        self.fading_mask = None
        self.fading_func = None
        self.fade_per_frame = 0
        self.last_update_time = 0
        self.ymargin = WIDTH * 3
        self.xmargin = round(self.ymargin * (screen.get_width() / screen.get_height()))
    def draw(self):
        left, bottom, width, height = self.rect
        if not layout.is_playing:
            pygame.draw.rect(self.subsurface, MARGIN, (0, 0, width, self.ymargin))
            pygame.draw.rect(self.subsurface, MARGIN, (0, 0, self.xmargin, height))
            pygame.draw.rect(self.subsurface, MARGIN, (width-self.xmargin, 0, self.xmargin, height))
            pygame.draw.rect(self.subsurface, MARGIN, (0, height-self.ymargin, width, self.ymargin))

        left += self.xmargin
        bottom += self.ymargin

        pos = layout.playing_index if layout.is_playing else movie.pos
        frame = movie.frame(pos).surface()
        screen.blit(movie.curr_bottom_layers_surface(pos, highlight=not layout.is_playing), (left, bottom), (0, 0, width, height))
        screen.blit(frame, (left, bottom), (0, 0, width, height))
        screen.blit(movie.curr_top_layers_surface(pos, highlight=not layout.is_playing), (left, bottom), (0, 0, width, height))

        if not layout.is_playing:
            mask = layout.timeline_area().combined_light_table_mask()
            if mask:
                screen.blit(mask, (left, bottom), (0, 0, width, height))
            if self.fading_mask:
                screen.blit(self.fading_mask, (left, bottom), (0, 0, width, height))

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
        frame = pygame.Surface((width-self.xmargin*2, height-self.ymargin*2), pygame.SRCALPHA)
        frame.fill(BACKGROUND)
        pg.surfarray.pixels_alpha(frame)[:] = 0
        pg.surfarray.pixels_alpha(frame)[:] = 0
        return frame

class TimelineArea:
    def __init__(self):
        # stuff for drawing the timeline
        self.frame_boundaries = []
        self.eye_boundaries = []
        self.prevx = None
        self.factors = [x*(0.75/0.6) for x in[0.7,0.6,0.5,0.4,0.3,0.2,0.15]]

        eye_icon_size = int(screen.get_width() * 0.15*0.14)
        self.eye_open = scale_image(pg.image.load('eye_open.png'), eye_icon_size)
        self.eye_shut = scale_image(pg.image.load('eye_shut.png'), eye_icon_size)

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

        # we can precombine the light table mask which doesn't change
        # unless we seek to a different frame, or the the definition of which
        # frames are on the light table changes
        self.combined_mask = None
        self.combined_on_light_table = None
        self.combined_movie_pos = None
        self.combined_movie_len = None

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

    def light_table_masks(self):
        masks = []
        for pos, color, transparency in self.light_table_positions():
            masks.append(movie.get_mask(pos, color, transparency))
        return masks

    def combined_light_table_mask(self):
        if movie.pos == self.combined_movie_pos and self.on_light_table == self.combined_on_light_table \
                and len(movie.frames) == self.combined_movie_len and self.combined_mask:
            return self.combined_mask
        self.combined_movie_pos = movie.pos
        self.combined_movie_len = len(movie.frames)
        self.combined_on_light_table = self.on_light_table.copy()
        
        masks = self.light_table_masks()
        if len(masks) == 0:
            self.combined_mask = None
        elif len(masks) == 1:
            self.combined_mask = masks[0]
        else:
            mask = masks[0].copy()
            alphas = []
            for m in masks[1:]:
                alphas.append(m.get_alpha())
                m.set_alpha(255) # TODO: this assumes the same transparency in all masks - might want to change
            mask.blits([(m, (0, 0), (0, 0, mask.get_width(), mask.get_height())) for m in masks[1:]])
            for m,a in zip(masks[1:],alphas):
                m.set_alpha(a)
            self.combined_mask = mask
        return self.combined_mask

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
        curr_frame_width = thumb_width(1)
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

class CachedLayerThumbnail:
    def __init__(self, layer, image):
        self.layer = layer
        self.image = image
        self.pos = movie.pos
        self.layer_pos = movie.layer_pos

    def valid(self):
        # don't cache the current layer. moving to another frame invalidates the cache. moving to another
        # layer does, too because we color the layers according to whether they're above or below the current layer
        return self.pos == movie.pos and self.layer_pos == movie.layer_pos and self.layer != movie.layer_pos

class LayersArea:
    def __init__(self):
        self.prevy = None
        self.thumbnails = {}
        self.cache_pos = None
        self.cache_layer_pos = None
        self.above_image = None
        self.below_image = None
    
    def cached_image(self, layer_pos, layer):
        if layer_pos not in self.thumbnails or not self.thumbnails[layer_pos].valid():
            left, bottom, width, height = self.rect
            image = scale_image(movie._blit_layers([layer], movie.pos), width)
            if layer_pos != movie.layer_pos: # color the image
                s = pg.Surface((image.get_width(), image.get_height()), pg.SRCALPHA)
                s.fill(BACKGROUND)
                if not self.above_image:
                    self.above_image = pg.Surface((image.get_width(), image.get_height()))
                    self.above_image.set_alpha(128)
                    self.above_image.fill(LAYERS_ABOVE)
                    self.below_image = pg.Surface((image.get_width(), image.get_height()))
                    self.below_image.set_alpha(128)
                    self.below_image.fill(LAYERS_BELOW)
                image.blit(self.above_image if layer_pos > movie.layer_pos else self.below_image, (0,0))
                image.set_alpha(128)
                s.blit(image, (0,0))
                image = s
            self.thumbnails[layer_pos] = CachedLayerThumbnail(layer_pos, image)
        return self.thumbnails[layer_pos].image

    def draw(self):
        left, bottom, width, height = self.rect
        top = bottom + width
        for layer_pos, layer in reversed(list(enumerate(movie.layers))):
            border = 1 + (layer_pos == movie.layer_pos)*2
            image = self.cached_image(layer_pos, layer)
            screen.blit(image, (left, bottom), image.get_rect()) 
            pygame.draw.rect(screen, PEN, (left, bottom, image.get_width(), image.get_height()), border)
            bottom += image.get_height()

    def new_delete_tool(self): return isinstance(layout.tool, NewDeleteTool)

    def y2frame(self, y):
        if not self.thumbnails or movie.layer_pos not in self.thumbnails or y is None:
            return None
        _, bottom, _, _ = self.rect
        return len(movie.layers) - ((y-bottom) // self.thumbnails[movie.layer_pos].image.get_height()) - 1

    def on_mouse_down(self,x,y):
        if self.new_delete_tool():
            if self.y2frame(y) == movie.layer_pos:
                layout.tool.layer_func()
                restore_tool() # we don't want multiple clicks in a row to delete lots of frames etc
            return
        f = self.y2frame(y)
        if f == movie.layer_pos:
            self.prevy = y
        else:
            self.prevy = None
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
        single_image_width = screen.get_width() * MOVIES_Y_SHARE
        for clipdir in get_clip_dirs():
            fulldir = os.path.join(WD, clipdir)
            with open(os.path.join(fulldir, CLIP_FILE), 'r') as clipfile:
                clip = json.loads(clipfile.read())
            curr_frame = clip['frame_pos']
            frame_file = os.path.join(fulldir, FRAME_FMT % curr_frame)
            self.images.append(scale_image(pg.image.load(frame_file), single_image_width))
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
                    image = scale_image(movie.curr_layers_surface(), image.get_width()) 
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

class TogglePlaybackButton:
    def __init__(self, play_icon, pause_icon):
        self.play = play_icon
        self.pause = pause_icon
        self.scaled = False
    def draw(self):
        left, bottom, width, height = self.rect
        if not self.scaled:
            self.play = scale_image(self.play, width, height)
            self.pause = scale_image(self.pause, width, height)
            self.scaled = True
            
        screen.blit(self.pause if layout.is_playing else self.play, (left, bottom), (0, 0, width, height))
    def on_mouse_down(self,x,y):
        toggle_playing()
    def on_mouse_up(self,x,y): pass
    def on_mouse_move(self,x,y): pass

Tool = collections.namedtuple('Tool', ['tool', 'cursor', 'chars'])

class LightTableMask:
    def __init__(self):
        self.surface = None
        self.color = None
        self.transparency = None
        self.movie_pos = None
        self.movie_len = None

class Thumbnail:
    def __init__(self):
        self.surface = None
        self.width = None
        self.height = None
        self.movie_pos = None
        self.movie_len = None

class Frame:
    def __init__(self, dir, layer_id=None, frame_id=None):
        self.dir = dir
        self.layer_id = layer_id
        if frame_id is not None: # id - load the surfaces from the directory
            self.id = frame_id
            for surf_id in self.surf_ids():
                setattr(self,surf_id,pygame.image.load(self.filename(surf_id)) if os.path.exists(self.filename(surf_id)) else None)
        else:
            self.id = str(uuid.uuid1())
            self.color = None
            self.lines = None

        self.dirty = False
        self.hold = False
        # we don't aim to maintain a "perfect" dirty flag such as "doing 5 things and undoing
        # them should result in dirty==False." The goal is to avoid gratuitous saving when
        # scrolling thru the timeline, which slows things down and prevents reopening
        # clips at the last actually-edited frame after exiting the program

    def _create_surfaces_if_needed(self):
        if self.color is not None:
            return
        self.color = layout.drawing_area().new_frame()
        self.lines = pg.Surface((self.color.get_width(), self.color.get_height()), pygame.SRCALPHA)
        self.lines.fill(PEN)
        pygame.surfarray.pixels_alpha(self.lines)[:] = 0
        self.dirty = True

    def surf_ids(self): return ['lines','color']
    def get_width(self): return empty_frame().color.get_width()
    def get_height(self): return empty_frame().color.get_height()
    def get_rect(self): return empty_frame().color.get_rect()

    def surf_by_id(self, surface_id):
        s = getattr(self, surface_id)
        return s if s is not None else empty_frame().surf_by_id(surface_id)

    def surface(self):
        if self.color is None:
            return empty_frame().color
        s = self.color.copy()
        s.blit(self.lines, (0, 0), (0, 0, s.get_width(), s.get_height()))
        return s

    def filename(self,surface_id):
        fname = f'{self.id}-{surface_id}.bmp'
        if self.layer_id:
            fname = os.path.join(f'layer-{self.layer_id}', fname)
        return os.path.join(self.dir, fname)
    def save(self):
        if self.dirty:
            for surf_id in self.surf_ids():
                pygame.image.save(self.surf_by_id(surf_id), self.filename(surf_id))
            self.dirty = False
    def delete(self):
        for surf_id in self.surf_ids():
            fname = self.filename(surf_id)
            if os.path.exists(fname):
                os.unlink(fname)

    def size(self):
        # a frame is 2 RGBA surfaces
        return (self.get_width() * self.get_height() * 8) if self.color is not None else 0

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
        for frame in frames:
            frame.layer_id = self.id
        subdir = self.subdir()
        if not os.path.isdir(subdir):
            os.makedirs(subdir)

    def frame(self, pos): # return the closest frame in the past where hold is false
        while self.frames[pos].hold:
            pos -= 1
        return self.frames[pos]

    def subdir(self): return os.path.join(self.dir, f'layer-{self.id}')
    def deleted_subdir(self): return self.subdir() + '-deleted'

    def delete(self): os.rename(self.subdir(), self.deleted_subdir())
    def undelete(self): os.rename(self.deleted_subdir(), self.subdir())

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

            self.layers = []
            for layer_index, layer_id in enumerate(layer_ids):
                frames = []
                for frame_index, frame_id in enumerate(frame_ids):
                    frame = Frame(dir, layer_id, frame_id)
                    frame.hold = holds[layer_index][frame_index]
                    frames.append(frame)
                self.layers.append(Layer(frames, dir, layer_id))

            self.pos = clip['frame_pos']
            self.layer_pos = clip['layer_pos']
            self.frames = self.layers[self.layer_pos].frames

        self.mask_cache = {}
        self.thumbnail_cache = {}

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

    def save_meta(self):
        clip = {
            'frame_pos':self.pos,
            'layer_pos':self.layer_pos,
            'frame_order':[frame.id for frame in self.frames],
            'layer_order':[layer.id for layer in self.layers],
            'hold':[[frame.hold for frame in layer.frames] for layer in self.layers],
        }
        text = json.dumps(clip,indent=2)
        with open(os.path.join(self.dir, CLIP_FILE), 'w') as clip_file:
            clip_file.write(text)

    def frame(self, pos):
        return self.layers[self.layer_pos].frame(pos)

    def get_mask(self, pos, color, transparency):
        assert pos != self.pos
        mask = self.mask_cache.setdefault(pos, LightTableMask())
        if pos != self.pos and mask.color == color and mask.transparency == transparency \
            and mask.movie_pos == self.pos and mask.movie_len == len(self.frames):
            return mask.surface
        mask.surface = pen2mask([layer.frame(pos).surf_by_id('lines') for layer in self.layers if layer.lit], color, transparency)
        mask.color = color
        mask.transparency = transparency
        mask.movie_pos = self.pos
        mask.movie_len = len(self.frames)
        return mask.surface

    def get_thumbnail(self, pos, width, height):
        thumbnail = self.thumbnail_cache.setdefault(pos, Thumbnail())
        # self.pos is "volatile" (being edited right now) - don't cache 
        if pos != self.pos and thumbnail.width == width and thumbnail.height == height and \
            thumbnail.movie_pos == self.pos and thumbnail.movie_len == len(self.frames):
            return thumbnail.surface
        thumbnail.movie_pos = self.pos
        thumbnail.movie_len = len(self.frames)
        thumbnail.width = width
        thumbnail.height = height
        # TODO: might be better to blit cached per-layer thumbnails than do this, especially important
        # for the current frame's thumbnail
        thumbnail.surface = scale_image(self._blit_layers(self.layers, pos), width, height)
        return thumbnail.surface

    def clear_cache(self):
        self.mask_cache = {}
        self.thumbnail_cache = {}
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

    def edit_curr_frame(self):
        f = self.frame(self.pos)
        f._create_surfaces_if_needed()
        f.dirty = True
        return f

    def _blit_layers(self, layers, pos, transparent=False):
        f = self.curr_frame()
        if transparent:
            s = pg.Surface((f.get_width(), f.get_height()), pg.SRCALPHA)
        else:
            s = make_surface(f.get_width(), f.get_height())
            s.fill(BACKGROUND)
        surfaces = []
        for layer in layers:
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
        # FIXME: cache this
        layers = self._blit_layers(self.layers[:self.layer_pos], pos, transparent=True)
        if not highlight:
            return layers
        layers.set_alpha(128)
        s = pg.Surface((layers.get_width(), layers.get_height()), pg.SRCALPHA)
        s.fill(BACKGROUND)
        if self.layer_pos == 0:
            return s
        below_image = pg.Surface((layers.get_width(), layers.get_height()), pg.SRCALPHA)
        below_image.set_alpha(128)
        below_image.fill(LAYERS_BELOW)
        alpha = pg.surfarray.array_alpha(layers)
        layers.blit(below_image, (0,0))
        pg.surfarray.pixels_alpha(layers)[:] = alpha
        self._set_undrawable_layers_grid(layers)
        s.blit(layers, (0,0))
        return s

    def curr_top_layers_surface(self, pos, highlight):
        layers = self._blit_layers(self.layers[self.layer_pos+1:], pos, transparent=True)
        if not highlight or self.layer_pos == len(self.layers)-1:
            return layers
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
        return s

    def curr_layers_surface(self):
        return self._blit_layers(self.layers, self.pos)

    def save_gif_and_pngs(self):
        with imageio.get_writer(self.dir + '.gif', fps=FRAME_RATE, mode='I') as writer:
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
        self.frame(self.pos).dirty = True # updates the image timestamp so we open at that image next time...
        self.frame(self.pos).save()
        self.save_gif_and_pngs()
        self.save_meta()
        self.garbage_collect_layer_dirs()

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
    def __init__(self, pos):
        self.pos = pos
    def undo(self):
        if movie.pos != self.pos:
            print('WARNING: wrong pos for a toggle-hold history item - expected {self.pos}, got {movie.pos}')
            movie.seek_frame(self.pos)
        movie.toggle_hold()
        layout.timeline_area().combined_mask = None
        return self
    def __str__(self):
        return f'ToggleHoldHistoryItem(toggling at frame {self.pos})'

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
    if movie.pos != 0:
        movie.toggle_hold()
        layout.timeline_area().combined_mask = None
        history.append_item(ToggleHoldHistoryItem(movie.pos))

TOOLS = {
    'pencil': Tool(PenTool(), pencil_cursor, 'bB'),
    'eraser': Tool(PenTool(BACKGROUND, WIDTH), eraser_cursor, 'eE'),
    'eraser-medium': Tool(PenTool(BACKGROUND, WIDTH*5), eraser_medium_cursor, 'rR'),
    'eraser-big': Tool(PenTool(BACKGROUND, WIDTH*20), eraser_big_cursor, 'tT'),
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
    'toggle-loop-mode': (toggle_loop_mode, 'l', None),
    'toggle-frame-hold': (toggle_frame_hold, 'h', None),
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
            #rgb = pygame.Color(0)
            #rgb.hsla = (i*10 % 360, 50, 50, 100)
            tool = Tool(PaintBucketTool(palette.colors[len(palette.colors)-row-1][col]), palette.cursors[len(palette.colors)-row-1][col], '')
            layout.add((x,y,color_w,color_w), ToolSelectionButton(tool))
            i += 1

    funcs_width = [
        ('insert-frame', 0.33),
        ('remove-frame', 0.33),
    ]
    offset = 0
    for func, width in funcs_width:
        #f, _, icon = FUNCTIONS[func]
        layout.add((offset*0.15,0.15,width*0.15, 0.1), ToolSelectionButton(TOOLS[func]))
        offset += width

    width = 0.05
    layout.add((0.1,0.15 + 0.1/2 - layout.aspect_ratio()*width/2,width, layout.aspect_ratio()*width), TogglePlaybackButton(pg.image.load('play.png'), pg.image.load('pause.png')))

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
        # TODO: pick the clip with the last-modified clip
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

    def __del__(self):
        for op in self.undo + self.redo:
            History.byte_size -= byte_size(op)

    def append_item(self, item):
        if nop(item):
            return

        self.undo.append(item)
        History.byte_size += byte_size(item) - sum([byte_size(op) for op in self.redo])
        self.redo = [] # forget the redo stack
        while self.undo and History.byte_size > MAX_HISTORY_BYTE_SIZE:
            History.byte_size -= byte_size(self.undo[0])
            del self.undo[0]

        layout.drawing_area().fading_mask = None # new operations invalidate old skeletons

    def undo_item(self):
        if self.undo:
            # TODO: we might want a loop here since some undo ops
            # turn out to be "no-ops" (specifically seek frame where we're already there.)
            # though we try to avoid having spurious seek-frame ops in the history
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

history = History()

escape = False

PLAYBACK_TIMER_EVENT = pygame.USEREVENT + 1
SAVING_TIMER_EVENT = pygame.USEREVENT + 2
FADING_TIMER_EVENT = pygame.USEREVENT + 3

pygame.time.set_timer(PLAYBACK_TIMER_EVENT, 1000//FRAME_RATE) # we play back at 12 fps
pygame.time.set_timer(SAVING_TIMER_EVENT, 15*1000) # we save a copy of the current clip every 15 seconds
pygame.time.set_timer(FADING_TIMER_EVENT, 1000//FADING_RATE) # we save a copy of the current clip every 15 seconds

interesting_events = [
    pygame.KEYDOWN,
    pygame.MOUSEMOTION,
    pygame.MOUSEBUTTONDOWN,
    pygame.MOUSEBUTTONUP,
    PLAYBACK_TIMER_EVENT,
    SAVING_TIMER_EVENT,
    FADING_TIMER_EVENT,
]

keyboard_shortcuts_enabled = False # enabled by Ctrl-A; disabled by default to avoid "surprises"
# upon random banging on the keyboard

set_tool(TOOLS['pencil'])

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
        
        if event.key == ord(' '):
            if pg.key.get_mods() & pg.KMOD_LCTRL:
                history.redo_item()
            else:
                history.undo_item()

        if keyboard_shortcuts_enabled:
          for tool in TOOLS.values():
            if event.key in [ord(c) for c in tool.chars]:
                set_tool(tool)

          for func, chars, _ in FUNCTIONS.values():
            if event.key in [ord(c) for c in chars]:
                func()
                
        if event.key == pygame.K_a and pygame.key.get_mods() & pygame.KMOD_CTRL:
            keyboard_shortcuts_enabled = not keyboard_shortcuts_enabled
            print('Ctrl-A pressed -','enabling' if keyboard_shortcuts_enabled else 'disabling','keyboard shortcuts')
      else:
          layout.on_event(event)

      # TODO: might be good to optimize repainting beyond "just repaint everything
      # upon every event"
      if layout.is_playing or (layout.drawing_area().fading_mask and event.type == FADING_TIMER_EVENT) or event.type not in [PLAYBACK_TIMER_EVENT, SAVING_TIMER_EVENT]:
        # don't repaint upon depressed mouse movement. this is important to avoid the pen
        # lagging upon "first contact" when a mouse motion event is sent before a mouse down
        # event at the same coordinate; repainting upon that mouse motion event loses time
        # when we should have been receiving the next x,y coordinates
        if event.type != pygame.MOUSEMOTION or layout.is_pressed:
            layout.draw()
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
