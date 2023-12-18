import pygame
import pygame.gfxdraw
import imageio
import winpath
import subprocess
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
from scipy.interpolate import splprep, splev

# this requires numpy to be installed in addition to scikit-image
from skimage.morphology import flood_fill, binary_dilation, skeletonize
pg = pygame

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Tinymate")
#screen = pygame.display.set_mode((500, 500))

FRAME_RATE = 12
PEN = (20, 20, 20)
BACKGROUND = (240, 235, 220)
UNDRAWABLE = (220, 215, 190)
WIDTH = 3 # the smallest width where you always have a pure pen color rendered along
# the line path, making our naive flood fill work well...
CURSOR_SIZE = int(screen.get_width() * 0.07)
MAX_HISTORY_BYTE_SIZE = 2*1024**3
FRAME_ORDER_FILE = 'frame_order.json'

MY_DOCUMENTS = winpath.get_my_documents()
WD = os.path.join(MY_DOCUMENTS if MY_DOCUMENTS else '.', 'Tinymate')
if not os.path.exists(WD):
    os.makedirs(WD)
print('clips read from, and saved to',WD)

import time
def image_from_fig(fig):
    start = time.time_ns()
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

def bspline_interp(points):
    x = np.array([1.*p[0] for p in points])
    y = np.array([1.*p[1] for p in points])

    okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
    x = np.r_[x[okay], x[-1]]#, x[0]]
    y = np.r_[y[okay], y[-1]]#, y[0]]

    def dist(i1, i2):
        return math.sqrt((x[i1]-x[i2])**2 + (y[i1]-y[i2])**2)
    curve_length = sum([dist(i, i+1) for i in range(len(x)-1)])
    bbox_length = (np.max(x)-np.min(x))*2 + (np.max(y)-np.min(y))*2
    endpoints_dist = dist(0, -1)

    make_closed = len(points)>2 and should_make_closed(curve_length, bbox_length, endpoints_dist)
    
    if make_closed:
        orig_len = len(x)
        def half(ls):
            ls = list(ls)
            return ls[:-len(ls)//2]
        x = np.array(list(x)+half([xi+0.001 for xi in x]))
        y = np.array(list(y)+half([yi+0.001 for yi in y]))

        tck, u = splprep([x, y], s=len(x)/5)

        ufirst = u[orig_len//2-1]
        ulast = u[-1]

    else: 
        tck, u = splprep([x, y], s=len(x)/5)

        ufirst = u[0]
        ulast = u[-1]

    step=(ulast-ufirst)/curve_length

    new_points = splev(np.arange(ufirst-step, ulast+step, step), tck)
    return new_points

def plotLines(points, ax, width):
    if len(set(points)) == 1:
        x,y = points[0]
        eps = 0.001
        points = [(x+eps, y+eps)] + points
    try:
        path = np.array(bspline_interp(points))
        px, py = path[0], path[1]
    except:
        px = np.array([x for x,y in points])
        py = np.array([y for x,y in points])
    ax.plot(py,px, linestyle='solid', color='k', linewidth=width, scalex=False, scaley=False, solid_capstyle='round')


def drawLines(image_height, image_width, points, width=3):
    global fig
    global ax
    if not fig:
        fig, ax = plt.subplots()
        ax.axis('off')
        # FIXME: need a reliable way to set the dimensions correctly
        fig.set_size_inches(image_width/fig.get_dpi()+0.01, image_height/fig.get_dpi())
        #fig.set_dpi(10)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    width *= 72 / fig.get_dpi()

    plt.cla()
    plt.xlim(0, image_width)
    plt.ylim(0, image_height)
    ax.invert_yaxis()
    ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
    ax.tick_params(left=False, right=False, bottom=False, top=False)

    plotLines(points, ax, width)

    return image_from_fig(fig)[:,:,0:3]

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
paint_bucket_cursor = load_cursor('paint_bucket.png', min_alpha=255)
blank_page_cursor = load_cursor('sheets.png', hot_spot=(0.5, 0.5))
garbage_bin_cursor = load_cursor('garbage.png', hot_spot=(0.5, 0.5))
pg.mouse.set_cursor(pencil_cursor[0])

class HistoryItem:
    def __init__(self, surface_id):
        self.surface_id = surface_id
        surface = movie.edit_curr_frame().surf_by_id(surface_id)
        self.saved_array = self.array(surface.copy())
        self.pos = movie.pos
        self.minx = 10**9
        self.miny = 10**9
        self.maxx = -10**9
        self.maxy = -10**9
        self.optimized = False
    def alpha(self):
        return self.surface_id == 'lines'
    def array(self,surface):
        return pg.surfarray.pixels_alpha(surface) if self.alpha() else pg.surfarray.pixels3d(surface)
    def undo(self):
        if self.pos != movie.pos:
            print(f'WARNING: HistoryItem at the wrong position! should be {self.pos}, but is {movie.pos}')
        movie.seek_frame(self.pos) # we should already be here, but just in case
        frame = self.array(movie.edit_curr_frame().surf_by_id(self.surface_id))
        if self.optimized:
            frame.blit(self.surface, (self.minx, self.miny), (0, 0, self.maxx-self.minx+1, self.maxy-self.miny+1))
        else:
            #frame.blit(self.surface, frame.get_rect())
            frame[:] = self.saved_array
    def affected(self,minx,miny,maxx,maxy):
        self.minx = min(minx,self.minx)
        self.maxx = max(maxx,self.maxx)
        self.miny = min(miny,self.miny)
        self.maxy = max(maxy,self.maxy)
    def optimize(self):
        return # FIXME
        if self.minx == 10**9:
            return
        left, bottom, width,height = movie.edit_curr_frame().get_rect()
        right = left+width
        top = bottom+height
        self.minx = max(self.minx, left)
        self.maxx = min(self.maxx, right-1)
        self.miny = max(self.miny, bottom)
        self.maxy = min(self.maxy, top-1)
        
        affected = make_surface(self.maxx-self.minx+1, self.maxy-self.miny+1)
        affected.blit(self.surface, (0,0), (self.minx, self.miny, self.maxx+1, self.maxy+1))
        self.surface = affected
        self.optimized = True

    def __str__(self):
        return f'HistoryItem(pos={self.pos}, rect=({self.minx}, {self.miny}, {self.maxx}, {self.maxy}))'

    def byte_size(self):
        width = self.saved_array.shape[0]
        height = self.saved_array.shape[1]
        return width*height*(1 if self.alpha() else 3)

class HistoryItemSet:
    def __init__(self, items):
        self.items = items
    def undo(self):
        for item in self.items:
            item.undo()
    def optimize(self):
        for item in self.items:
            item.optimize()
    def byte_size(self):
        return sum([item.byte_size() for item in self.items])

class PenTool:
    def __init__(self, eraser=False, width=WIDTH):
        self.prev_drawn = None
        self.color = BACKGROUND if eraser else PEN
        self.eraser = eraser
        self.width = width
        self.circle_width = (width//2)*2
        self.history_item = None
        self.points = []
        self.lines_array = None

    def draw(self, rect, cursor_surface):
        left, bottom, width, height = rect
        _, _, w, h = cursor_surface.get_rect()
        scaled_width = w*height/h
        surface = scale_image(cursor_surface, scaled_width, height)
        screen.blit(surface, (left+width/2-scaled_width/2, bottom), (0, 0, scaled_width, height))

    def on_mouse_down(self, x, y):
        self.points = []
        self.bucket_color = None
        self.lines_array = pg.surfarray.pixels_alpha(movie.edit_curr_frame().surf_by_id('lines'))
        self.on_mouse_move(x,y)

    def on_mouse_up(self, x, y):
        self.lines_array = None
        self.history_item = HistoryItem('lines')
        start = time.time_ns()
        self.points.append((x,y))
        self.prev_drawn = None
        frame = movie.edit_curr_frame().surf_by_id('lines')

        new_lines = drawLines(frame.get_width(), frame.get_height(), self.points, self.width)
        lines = pygame.surfarray.pixels_alpha(frame)
        if self.eraser:
            lines[:,:] = np.minimum(new_lines[:,:,0], lines[:,:])
        else:
            lines[:,:] = np.maximum(255-new_lines[:,:,0], lines[:,:])

        if self.eraser:
            color_history_item = HistoryItem('color')
            color = pg.surfarray.pixels3d(movie.edit_curr_frame().surf_by_id('color'))
            flood_fill_color_based_on_lines(color, lines, x, y, self.bucket_color if self.bucket_color else BACKGROUND)
            self.history_item = HistoryItemSet([self.history_item, color_history_item])

        if self.history_item:
            self.history_item.optimize()
            history_append(self.history_item)
            self.history_item = None

    def on_mouse_move(self, x, y):
       if self.eraser and self.bucket_color is None and self.lines_array[x,y] != 255:
           self.bucket_color = movie.edit_curr_frame().surf_by_id('color').get_at((x,y))
       draw_into = screen.subsurface(layout.drawing_area().rect)
       self.points.append((x,y))
       if self.history_item:
            self.history_item.affected(x-self.width,y-self.width,x+self.width,y+self.width)
       color = self.color if not self.eraser else (self.bucket_color if self.bucket_color else BACKGROUND)
       if self.prev_drawn:
            drawLine(draw_into, self.prev_drawn, (x,y), color, self.width)
       drawCircle(draw_into, x, y, color, self.circle_width)
       self.prev_drawn = (x,y) 

class NewDeleteTool(PenTool):
    def __init__(self, frame_func, clip_func):
        self.frame_func = frame_func
        self.clip_func = clip_func

    def on_mouse_down(self, x, y): pass
    def on_mouse_up(self, x, y): pass
    def on_mouse_move(self, x, y): pass

def flood_fill_color_based_on_lines(color, lines, x, y, bucket_color):
    pen_mask = lines == 255
    flood_code = 2
    t1=time.time_ns()
    flood_mask = flood_fill(pen_mask.astype(np.byte), (x,y), flood_code) == flood_code
    t2=time.time_ns()
    #flood_mask = binary_dilation(skeletonize(flood_mask,method='lee'))
    #t3=time.time_ns()
    #print('flood',(t2-t1)/10**6,'skeleton',(t3-t2)/10**6)
    for ch in range(3):
         color[:,:,ch] = color[:,:,ch]*(1-flood_mask) + bucket_color[ch]*flood_mask

class PaintBucketTool:
    def __init__(self,color):
        self.color = color
    def draw(self, rect, cursor_surface):
        left, bottom, width, height = rect
        x = left + width//2
        y = bottom + height//2
        rx = int(0.85*width)//2
        ry = int(0.9*height)//2
        pygame.gfxdraw.filled_ellipse(screen, x, y, rx, ry, self.color)
        pygame.gfxdraw.aaellipse(screen, x, y, rx, ry, PEN)
        pygame.gfxdraw.aaellipse(screen, x, y, rx-1, ry-1, PEN)
    def on_mouse_down(self, x, y):
        color = pygame.surfarray.pixels3d(movie.edit_curr_frame().surf_by_id('color'))
        lines = pygame.surfarray.pixels_alpha(movie.edit_curr_frame().surf_by_id('lines'))
        
        if np.array_equal(color[x,y,:], np.array(self.color)) or lines[x,y] == 255:
            return # we never flood the lines themselves - they keep the PEN color in a separate layer;
            # and there's no point in flooding with the color the pixel already has

        # TODO: would be better to optimize the history item
        history_append(HistoryItem('color'))

        flood_fill_color_based_on_lines(color, lines, x, y, self.color)
        
    def on_mouse_up(self, x, y):
        pass
    def on_mouse_move(self, x, y):
        pass

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
        self.elems.append(elem)

    def draw(self):
        if self.is_pressed and self.focus_elem is self.drawing_area():
            return
        screen.fill(UNDRAWABLE)
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
            else:
                return

        if event.type == SAVING_TIMER_EVENT:
            movie.frame(movie.pos).save()
            return

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
            
def pen2mask(lines, rgb, transparency):
    mask_surface = pygame.Surface((lines.get_width(), lines.get_height()), pygame.SRCALPHA)
    mask = pygame.surfarray.pixels3d(mask_surface)
    pen = pygame.surfarray.pixels_alpha(lines)
    for ch in range(3):
        mask[:,:,ch] = rgb[ch]
    pygame.surfarray.pixels_alpha(mask_surface)[:] = pen
    mask_surface.set_alpha(int(transparency*255))
    return mask_surface

class DrawingArea:
    def __init__(self):
        pass
    def draw(self):
        left, bottom, width, height = self.rect
        frame = movie.frame(layout.playing_index).surface() if layout.is_playing else movie.curr_frame().surface()
        screen.blit(frame, (left, bottom), (0, 0, width, height))

        if not layout.is_playing:
            mask = layout.timeline_area().combined_light_table_mask()
            if mask:
                screen.blit(mask, (left, bottom), (0, 0, width, height))

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
        frame = make_surface(width, height)
        frame.fill(BACKGROUND)
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
            color = (0,0,brightness) if pos_dist < 0 else (0,brightness,0)
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

def get_last_modified(filenames):
    f2mtime = {}
    for f in filenames:
        s = os.stat(f)
        f2mtime[f] = s.st_mtime
    return list(sorted(f2mtime.keys(), key=lambda f: f2mtime[f]))[-1]

class MovieListArea:
    def __init__(self):
        self.show_pos = None
        self.prevy = None
        self.reload()
    def reload(self):
        self.clips = []
        self.images = []
        for clipdir in get_clip_dirs():
            clip = os.path.join(WD, clipdir)
            try:
                ids_and_names = clip_frame_filenames(clip)
                file2id = dict([(f, frameid) for frameid, f, h in ids_and_names])
                last_modified = get_last_modified([f for frameid, f, h in ids_and_names])
                last_modified_id = file2id[last_modified]
            except:
                continue
            self.clips.append(clip)
            self.images.append(scale_image(Frame(last_modified_id, clip).surface(), int(screen.get_width() * 0.15)))
        self.clip_pos = 0 
    def draw(self):
        left, bottom, width, height = self.rect
        first = True
        pos = self.show_pos if self.show_pos is not None else self.clip_pos
        for image in self.images[pos:]:
            border = 1 + first*2
            if first and pos == self.clip_pos:
                try:
                    image = scale_image(movie.curr_frame().surface(), image.get_width()) 
                    self.images[pos] = image # this keeps the image correct when scrolled out of clip_pos
                    # (we don't self.reload() upon scrolling so self.images can go stale when the current
                    # clip is modified)
                except:
                    pass
            first = False
            screen.blit(image, (left, bottom), image.get_rect()) 
            pygame.draw.rect(screen, PEN, (left, bottom, image.get_width(), image.get_height()), border)
            bottom += image.get_height()
    def new_delete_tool(self): return isinstance(layout.tool, NewDeleteTool) 
    def y2frame(self, y):
        if not self.images or y is None:
            return None
        return y // self.images[0].get_height()
    def on_mouse_down(self,x,y):
        if self.new_delete_tool():
            # TODO: add condition on position
            layout.tool.clip_func()
            restore_tool()
            return
        self.prevy = y
        self.show_pos = self.clip_pos
    def on_mouse_move(self,x,y):
        if self.prevy is None:
            self.prevy = y # this happens eg when a new_delete_tool is used upon mouse down
            # and then the original tool is restored
            self.show_pos = self.clip_pos
        if self.new_delete_tool():
            return
        prev_pos = self.y2frame(self.prevy)
        curr_pos = self.y2frame(y)
        if prev_pos is None and curr_pos is None:
            self.prevy = y
            return
        if curr_pos is not None and prev_pos is not None:
            pos_dist = prev_pos - curr_pos
        else:
            pos_dist = -1 if y > self.prevy else 1
        self.prevy = y
        self.show_pos = min(max(0, self.show_pos + pos_dist), len(self.clips)-1) 
    def on_mouse_up(self,x,y):
        self.on_mouse_move(x,y)
        # opening a movie is a slow operation so we don't want it to be "too interactive"
        # (like timeline scrolling) - we wait for the mouse-up event to actually open the clip
        self.open_clip(self.show_pos)
        self.prevy = None
        self.show_pos = None
    def open_clip(self, clip_pos):
        if clip_pos == self.clip_pos:
            return
        global movie
        movie.save_before_closing()
        movie = Movie(self.clips[clip_pos])
        self.clip_pos = clip_pos

class ToolSelectionButton:
    def __init__(self, tool):
        self.tool = tool
    def draw(self):
        self.tool.tool.draw(self.rect,self.tool.cursor[1])
    def on_mouse_down(self,x,y):
        set_tool(self.tool)
    def on_mouse_up(self,x,y): pass
    def on_mouse_move(self,x,y): pass

class FunctionButton:
    def __init__(self, function, icon=None):
        self.function = function
        self.icon = icon
        self.scaled = None
    def draw(self):
        # TODO: show it was pressed (tool selection button shows it by changing the cursor, maybe still should show it was pressed)
        if self.icon:
            left, bottom, width, height = self.rect
            if not self.scaled:
                self.scaled = scale_image(self.icon, width, height)
            screen.blit(self.scaled, (left, bottom), (0, 0, width, height))
    def on_mouse_down(self,x,y):
        self.function()
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
    def __init__(self, color_surface_or_id, dir):
        self.dir = dir
        if type(color_surface_or_id) == str: # id - load the surfaces from the directory
            self.id = color_surface_or_id
            for surf_id in self.surf_ids():
                setattr(self,surf_id,pygame.image.load(self.filename(surf_id)))
            self.dirty = False
        else:
            self.color = color_surface_or_id
            blank = make_surface(color_surface_or_id.get_width(), color_surface_or_id.get_height())
            self.lines = pygame.Surface((blank.get_width(), blank.get_height()), pygame.SRCALPHA, blank.copy())
            self.lines.fill(PEN)
            pygame.surfarray.pixels_alpha(self.lines)[:,:] = 0
            self.id = str(uuid.uuid1())
            self.dirty = True

        self.hold = False
        # we don't aim to maintain a "perfect" dirty flag such as "doing 5 things and undoing
        # them should result in dirty==False." The goal is to avoid gratuitous saving when
        # scrolling thru the timeline, which slows things down and prevents reopening
        # clips at the last actually-edited frame after exiting the program

    def surf_ids(self): return ['lines','color']
    def get_width(self): return self.lines.get_width()
    def get_height(self): return self.lines.get_height()
    def get_rect(self): return self.lines.get_rect()

    def surf_by_id(self, surface_id): return getattr(self, surface_id)

    def surface(self):
        s = self.color.copy()
        s.blit(self.lines, (0, 0), (0, 0, s.get_width(), s.get_height()))
        return s

    def filename(self,surface_id): return os.path.join(self.dir, f'{self.id}-{surface_id}.bmp')
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

def clip_frame_filenames(clipdir):
    with open(os.path.join(clipdir, FRAME_ORDER_FILE), 'r') as frame_order:
        frames = json.loads(frame_order.read())
        # we only return the lines layer filename per frame - since we always write out both lines and color,
        # this is good enough for checking timestamps
        return [(frame['id'], os.path.join(clipdir, frame['id']+'-lines.bmp'), frame['hold']) for frame in frames]

class Movie:
    def __init__(self, dir):
        self.dir = dir
        if not os.path.isdir(dir): # new clip
            os.makedirs(dir)
            self.frames = [Frame(layout.drawing_area().new_frame(), self.dir)]
            self.frames[0].save()
            self.pos = 0
            self.save_meta()
        else:
            self.frames = []
            fname2id = {}
            for frameid, fname, hold in clip_frame_filenames(self.dir):
                fname2id[fname] = frameid
                frame = Frame(frameid, self.dir)
                frame.hold = hold
                self.frames.append(frame)
            last_modified_id = fname2id[get_last_modified(fname2id.keys())]
            # reopen at the last modified frame
            self.pos = [frame.id for frame in self.frames].index(last_modified_id)
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

    def _surface_pos(self, pos):
        while self.frames[pos].hold:
            pos -= 1
        return pos

    def frame(self, pos): # return the closest frame in the past where hold is false
        return self.frames[self._surface_pos(pos)]

    def save_meta(self):
        frames = [{'id':frame.id,'hold':frame.hold} for frame in self.frames]
        text = json.dumps(frames,indent=2)
        with open(os.path.join(self.dir, FRAME_ORDER_FILE), 'w') as frame_order:
            frame_order.write(text)

    def get_mask(self, pos, color, transparency):
        assert pos != self.pos
        pos = self._surface_pos(pos)
        mask = self.mask_cache.setdefault(pos, LightTableMask())
        if pos != self.pos and mask.color == color and mask.transparency == transparency \
            and mask.movie_pos == self.pos and mask.movie_len == len(self.frames):
            return mask.surface
        mask.surface = pen2mask(self.frames[pos].surf_by_id('lines'), color, transparency)
        mask.color = color
        mask.transparency = transparency
        mask.movie_pos = self.pos
        mask.movie_len = len(self.frames)
        return mask.surface

    def get_thumbnail(self, pos, width, height):
        pos = self._surface_pos(pos)
        thumbnail = self.thumbnail_cache.setdefault(pos, Thumbnail())
        # self.pos is "volatile" (being edited right now) - don't cache 
        if pos != self.pos and thumbnail.width == width and thumbnail.height == height and \
            thumbnail.movie_pos == self.pos and thumbnail.movie_len == len(self.frames):
            return thumbnail.surface
        thumbnail.movie_pos = self.pos
        thumbnail.movie_len = len(self.frames)
        thumbnail.width = width
        thumbnail.height = height
        thumbnail.surface = scale_image(self.frames[pos].surface(), width, height)
        return thumbnail.surface

    def clear_cache(self):
        self.mask_cache = {}
        self.thumbnail_cache = {}

    def seek_frame(self,pos):
        assert pos >= 0 and pos < len(self.frames)
        self.frame(self.pos).save()
        self.pos = pos
        self.clear_cache()
        self.save_meta()

    def next_frame(self): self.seek_frame((self.pos + 1) % len(self.frames))
    def prev_frame(self): self.seek_frame((self.pos - 1) % len(self.frames))

    def insert_frame(self):
        self.frames.insert(self.pos+1, Frame(layout.drawing_area().new_frame(), self.dir))
        self.next_frame()

    def insert_frame_at_pos(self, pos, frame):
        assert pos >= 0 and pos <= len(self.frames)
        self.frame(self.pos).save()
        self.pos = pos
        self.frames.insert(self.pos, frame)
        self.clear_cache()
        frame.save()
        self.save_meta()

    # TODO: this works with pos modified from the outside but it's scary as the API
    def remove_frame(self, at_pos=-1, new_pos=-1):
        if len(self.frames) <= 1:
            return

        self.clear_cache()

        if at_pos == -1:
            at_pos = self.pos
        else:
            self.frame(self.pos).save()
        self.pos = at_pos

        removed = self.frames[self.pos]

        del self.frames[self.pos]
        removed.delete()

        self.frames[0].hold = False # could have been made true if we deleted frame 0
        # and frame 1 had hold==True - now this wouldn't make sense

        if self.pos >= len(self.frames):
            self.pos = len(self.frames)-1

        if new_pos >= 0:
            self.pos = new_pos

        self.save_meta()

        return removed

    def curr_frame(self):
        return self.frame(self.pos)

    def edit_curr_frame(self):
        f = self.frame(self.pos)
        f.dirty = True
        return f

    def save_gif(self):
        with imageio.get_writer(self.dir + '.gif', fps=FRAME_RATE, mode='I') as writer:
            for i in range(len(self.frames)):
                frame = self.frame(i)
                writer.append_data(np.transpose(pygame.surfarray.pixels3d(frame.surface()), [1,0,2]))

    def save_before_closing(self):
        history_clear()
        self.frame(self.pos).dirty = True # updates the image timestamp so we open at that image next time...
        self.frame(self.pos).save()
        self.save_gif()
        self.save_meta()

class SeekFrameHistoryItem:
    def __init__(self, pos): self.pos = pos
    def undo(self): movie.seek_frame(self.pos) 
    def __str__(self): return f'SeekFrameHistoryItem(restoring pos to {self.pos})'

class InsertFrameHistoryItem:
    def __init__(self, pos): self.pos = pos
    def undo(self):
        # normally remove_frame brings you to the next frame after the one you removed.
        # but when undoing insert_frame, we bring you to the previous frame after the one
        # you removed - it's the one where you inserted the frame we're now removing to undo
        # the insert, so this is where we should go to bring you back in time.
        movie.remove_frame(at_pos=self.pos, new_pos=max(0, self.pos-1))
    def __str__(self):
        return f'InsertFrameHistoryItem(removing at pos {self.pos}, then seeking to pos {(self.pos-1)%len(movie.frames)})'

class RemoveFrameHistoryItem:
    def __init__(self, pos, frame, next_hold):
        self.pos = pos
        self.frame = frame
        frame.dirty = True # otherwise undo() will not save the frame to disk... which is bad
        # because remove_frame() deleted it from the disk!
        self.next_hold = next_hold
    def undo(self):
        movie.insert_frame_at_pos(self.pos, self.frame)
        # there's a special case when removing frame 0 while frames[1].hold was True -
        # remove_frame will force it to False, in that case we need to restore it
        if self.next_hold and self.pos==0:
            movie.frames[1].hold = True
            movie.save_meta()
    def __str__(self):
        return f'RemoveFrameHistoryItem(inserting at pos {self.pos})'
    def byte_size(self):
        f = self.frame
        return f.get_width()*f.get_height()*4

class ToggleHoldHistoryItem:
    def __init__(self, pos):
        self.pos = pos
    def undo(self):
        if movie.pos != self.pos:
            print('WARNING: wrong pos for a toggle-hold history item - expected {self.pos}, got {movie.pos}')
            movie.seek_frame(self.pos)
        movie.toggle_hold()
        layout.timeline_area().combined_mask = None

def append_seek_frame_history_item_if_frame_is_dirty():
    if history and not isinstance(history[-1], SeekFrameHistoryItem):
        history_append(SeekFrameHistoryItem(movie.pos))

def insert_frame():
    movie.insert_frame()
    history_append(InsertFrameHistoryItem(movie.pos))

def remove_frame():
    if len(movie.frames) == 1:
        return
    pos = movie.pos
    next_hold = False if movie.pos+1 == len(movie.frames) else movie.frames[movie.pos+1].hold
    removed = movie.remove_frame()
    history_append(RemoveFrameHistoryItem(pos, removed, next_hold))

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
    layout.movie_list_area().reload()

def remove_clip():
    movie_list_area = layout.movie_list_area()
    if len(movie_list_area.clips) <= 1:
        return # we don't remove the last clip - if we did we'd need to create a blank one,
        # which is a bit confusing. [we can't remove the last frame in a timeline, either]
    global movie
    movie.save_before_closing()
    os.rename(movie.dir, movie.dir + '-deleted')
    movie_list_area.reload()

    movie = Movie(movie_list_area.clips[0])

def toggle_playing(): layout.toggle_playing()

def toggle_loop_mode():
    timeline_area = layout.timeline_area()
    timeline_area.loop_mode = not timeline_area.loop_mode

def toggle_frame_hold():
    if movie.pos != 0:
        movie.toggle_hold()
        layout.timeline_area().combined_mask = None
        history_append(ToggleHoldHistoryItem(movie.pos))

TOOLS = {
    'pencil': Tool(PenTool(), pencil_cursor, 'bB'),
    'eraser': Tool(PenTool(BACKGROUND, WIDTH), eraser_cursor, 'eE'),
    'eraser-medium': Tool(PenTool(BACKGROUND, WIDTH*5), eraser_medium_cursor, 'rR'),
    'eraser-big': Tool(PenTool(BACKGROUND, WIDTH*20), eraser_big_cursor, 'tT'),
    # insert/remove frame are both a "tool" (with a special cursor) and a "function."
    # meaning, when it's used thru a keyboard shortcut, a frame is inserted/removed
    # without any more ceremony. but when it's used thru a button, the user needs to
    # actually press on the current image in the timeline to remove/insert. this,
    # to avoid accidental removes/inserts thru misclicks and a resulting confusion
    # (a changing cursor is more obviously "I clicked a wrong button, I should click
    # a different one" than inserting/removing a frame where you need to undo but to
    # do that, you need to understand what just happened)
    'insert-frame': Tool(NewDeleteTool(insert_frame, insert_clip), blank_page_cursor, ''),
    'remove-frame': Tool(NewDeleteTool(remove_frame, remove_clip), garbage_bin_cursor, ''),
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
        pg.mouse.set_cursor(tool.cursor[0])

def restore_tool():
    set_tool(prev_tool)

def color_image(s, color):
    dr, dg, db = color
    sc = s.copy()
    for y in range(s.get_height()):
        for x in range(s.get_width()):
            r,g,b,a = s.get_at((x,y))
            new = (int(r*dr/255), int(g*dg/255), int(b*db/255))
            sc.set_at((x,y), new+(a,))
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
        colors[0] = [BACKGROUND, (192, 192, 192), white]
        color2popularity = dict(list(reversed(sorted(list(color_hist.items()), key=lambda x: x[1])))[:(rows-1)*columns])
        hit2color = [(first_hit, color) for color, first_hit in sorted(list(first_color_hit.items()), key=lambda x: x[1])]

        row = 1
        col = 0
        for hit, color in hit2color:
            if color in color2popularity:
                colors[row][col] = color
                row+=1
                if row == rows:
                    row = 1
                    col += 1

        self.rows = rows
        self.columns = columns
        self.colors = colors

        self.init_cursors()

    def init_cursors(self):
        global paint_bucket_cursor
        s = paint_bucket_cursor[1]
        self.cursors = [[None for col in range(self.columns)] for row in range(self.rows)]
        for row in range(self.rows):
            for col in range(self.columns):
                sc = color_image(s, self.colors[row][col])
                self.cursors[row][col] = (pg.cursors.Cursor((0,sc.get_height()-1), sc), sc)


palette = Palette('palette.png')

def get_clip_dirs():
    '''returns the clip directories sorted by last modification time (latest first)'''
    wdfiles = os.listdir(WD)
    clipdirs = {}
    for d in wdfiles:
        try:
            if d.endswith('-deleted'):
                continue
            frame_order_file = os.path.join(os.path.join(WD, d), FRAME_ORDER_FILE)
            s = os.stat(frame_order_file)
            clipdirs[d] = s.st_mtime
        except:
            continue

    return list(reversed(sorted(clipdirs.keys(), key=lambda d: clipdirs[d])))

def init_layout_basic():
    screen.fill(UNDRAWABLE)

    global layout
    layout = Layout()
    layout.add((0.15,0.15,0.7,0.85), DrawingArea())

def init_layout_rest():
    layout.add((0, 0, 1, 0.15), TimelineArea())
    layout.add((0.85, 0.15, 0.15, 0.85), MovieListArea())

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
    
    for row,y in enumerate(np.arange(0.25,0.85-0.001,color_w)):
        for col,x in enumerate(np.arange(0,0.15-0.001,color_w)):            
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
        layout.add((offset*0.15,0.15,width*0.15, 0.1), ToolSelectionButton(TOOLS[func]))#FunctionButton(f, icon))
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

# The history is "global" for all operations. In some (rare) animation programs
# there's a history per frame. One problem with this is how to undo timeline
# operations like frame deletions (do you have a separate undo function for this?)
# It's also somewhat less intuitive in that you might have long forgotten
# what you've done on some frame when you visit it and press undo one time
# too many
history = []
history_byte_size = 0

def byte_size(history_item):
    return getattr(history_item, 'byte_size', lambda: 128)()

def history_append(item):
    global history_byte_size
    history_byte_size += byte_size(item)
    history.append(item)
    while history and history_byte_size > MAX_HISTORY_BYTE_SIZE:
        history_byte_size -= byte_size(history[0])
        del history[0]

def history_pop():
    global history_byte_size
    if history:
        # TODO: we might want a loop here since some undo ops
        # turn out to be "no-ops" (specifically seek frame where we're already there.)
        # as it is right now, you might press undo and nothing will happen which might be confusing
        last_op = history[-1]
        last_op.undo()
        history_byte_size -= byte_size(last_op)
        history.pop()

def history_clear():
    global history
    global history_byte_size
    history = []
    history_byte_size = 0

escape = False

PLAYBACK_TIMER_EVENT = pygame.USEREVENT + 1
SAVING_TIMER_EVENT = pygame.USEREVENT + 2

pygame.time.set_timer(PLAYBACK_TIMER_EVENT, 1000//FRAME_RATE) # we play back at 12 fps
pygame.time.set_timer(SAVING_TIMER_EVENT, 15*1000) # we save a copy of the current clip every 15 seconds

interesting_events = [
    pygame.KEYDOWN,
    pygame.MOUSEMOTION,
    pygame.MOUSEBUTTONDOWN,
    pygame.MOUSEBUTTONUP,
    PLAYBACK_TIMER_EVENT,
    SAVING_TIMER_EVENT,
]

keyboard_shortcuts_enabled = False # enabled by Ctrl-A; disabled by default to avoid "surprises"
# upon random banging on the keyboard

# add tdiff() to printouts to see how many ms passed since the last call to tdiff()
prevts=time.time_ns()
def tdiff():
    global prevts
    now=time.time_ns()
    diff=(now-prevts)//10**6
    prevts = now
    return diff

while not escape: 
 try:
  for event in pygame.event.get():
   if event.type not in interesting_events:
       continue
   try:
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE: # ESC pressed
            escape = True
            break

        if layout.is_pressed:
            continue # ignore keystrokes (except ESC) when a mouse tool is being used
        
        if event.key == ord(' '): # undo
            history_pop()

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
      if layout.is_playing or event.type not in [PLAYBACK_TIMER_EVENT, SAVING_TIMER_EVENT]:
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
