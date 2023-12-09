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
import sys
import os
import json

# this requires numpy to be installed in addition to scikit-image
from skimage.morphology import flood_fill
pg = pygame

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Tinymate")
#screen = pygame.display.set_mode((500, 500))

FRAME_RATE = 12
PEN = (20, 20, 20)
BACKGROUND = (240, 235, 220)
UNDRAWABLE = (220, 215, 190)
WIDTH = 5 
CURSOR_SIZE = int(screen.get_width() * 0.07)
SCALE = 1
MAX_HISTORY_BYTE_SIZE = 2*1024**3
FRAME_ORDER_FILE = 'frame_order.json'

MY_DOCUMENTS = winpath.get_my_documents()
WD = os.path.join(MY_DOCUMENTS if MY_DOCUMENTS else '.', 'Tinymate')
if not os.path.exists(WD):
    os.makedirs(WD)
print('clips read from, and saved to',WD)


def drawCircle( screen, x, y, color, width):
  pygame.draw.circle( screen, color, ( x, y ), width/2 )

def drawLine(screen, pos1, pos2, color, width):
    if True or width != 5:
        pygame.draw.line(screen, color, pos1, pos2, width)
        return
    def oft(p,x,y):
        return p[0]+x, p[1]+y
    pygame.draw.aaline(screen, color, pos1, pos2)
    pygame.draw.aaline(screen, color, oft(pos1,-1,-1), oft(pos2,-1,-1))
    pygame.draw.aaline(screen, color, oft(pos1,-1,1), oft(pos2,-1,1))
    pygame.draw.aaline(screen, color, oft(pos1,1,1), oft(pos2,1,1))
    pygame.draw.aaline(screen, color, oft(pos1,1,-1), oft(pos2,1,-1))
    pygame.draw.aaline(screen, color, oft(pos1,1,0), oft(pos2,1,0))
    pygame.draw.aaline(screen, color, oft(pos1,-1,0), oft(pos2,-1,0))
    pygame.draw.aaline(screen, color, oft(pos1,0,1), oft(pos2,0,1))
    pygame.draw.aaline(screen, color, oft(pos1,0,-1), oft(pos2,0,-1))
    pygame.draw.aaline(screen, color, oft(pos1,2,0), oft(pos2,2,0))
    pygame.draw.aaline(screen, color, oft(pos1,-2,0), oft(pos2,-2,0))
    pygame.draw.aaline(screen, color, oft(pos1,0,2), oft(pos2,0,2))
    pygame.draw.aaline(screen, color, oft(pos1,0,-2), oft(pos2,0,-2))

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
    def __init__(self):
        self.surface = movie.edit_curr_frame().copy()
        self.pos = movie.pos
        self.minx = 10**9
        self.miny = 10**9
        self.maxx = -10**9
        self.maxy = -10**9
        self.optimized = False
    def undo(self):
        if self.pos != movie.pos:
            print(f'WARNING: HistoryItem at the wrong position! should be {self.pos}, but is {movie.pos}')
        movie.seek_frame(self.pos) # we should already be here, but just in case
        frame = movie.edit_curr_frame()
        if self.optimized:
            frame.blit(self.surface, (self.minx, self.miny), (0, 0, self.maxx-self.minx+1, self.maxy-self.miny+1))
        else:
            frame.blit(self.surface, frame.get_rect())
    def affected(self,minx,miny,maxx,maxy):
        self.minx = min(minx,self.minx)
        self.maxx = max(maxx,self.maxx)
        self.miny = min(miny,self.miny)
        self.maxy = max(maxy,self.maxy)
    def optimize(self):
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
        width = self.surface.get_width()
        height = self.surface.get_height()
        return width*height*4 # assuming 8b RGBA

class PenTool:
    def __init__(self, color=PEN, width=WIDTH):
        self.prev_drawn = None
        self.color = color
        self.width = width*SCALE
        self.circle_width = (width//2)*2*SCALE
        self.history_item = None

    def draw(self, rect, cursor_surface):
        left, bottom, width, height = rect
        _, _, w, h = cursor_surface.get_rect()
        scaled_width = w*height/h
        surface = scale_image(cursor_surface, scaled_width, height)
        screen.blit(surface, (left+width/2-scaled_width/2, bottom), (0, 0, scaled_width, height))

    def on_mouse_down(self, x, y):
        self.history_item = HistoryItem()
        frame = movie.edit_curr_frame()
        drawCircle(frame, x, y, self.color, self.circle_width)
        self.on_mouse_move(x,y)

    def on_mouse_up(self, x, y):
        self.prev_drawn = None
        frame = movie.edit_curr_frame()
        drawCircle(frame, x, y, self.color, self.circle_width)
        if self.history_item:
            self.history_item.optimize()
            history_append(self.history_item)
            self.history_item = None

    def on_mouse_move(self, x, y):
       if self.history_item:
            self.history_item.affected(x-self.width,y-self.width,x+self.width,y+self.width)
       frame = movie.edit_curr_frame()
       if self.prev_drawn:
            drawLine(frame, self.prev_drawn, (x,y), self.color, self.width)
       drawCircle(frame, x, y, self.color, self.circle_width)
       self.prev_drawn = (x,y)

class NewDeleteTool(PenTool):
    def __init__(self, frame_func, clip_func):
        self.frame_func = frame_func
        self.clip_func = clip_func

    def on_mouse_down(self, x, y): pass
    def on_mouse_up(self, x, y): pass
    def on_mouse_move(self, x, y): pass

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
        # TODO: would be better to optimize the history item
        surface = movie.edit_curr_frame()
        xy_color = surface.get_at((x,y))
        if xy_color == PEN or xy_color == self.color:
            return # never flood pen lines or the light table won't work properly (meaning,
            # don't create an illusion that flooding pen lines is a nice way to get colored lines.
            # of course nothing prevents someone from figuring out that the eraser can be used as a pencil
            # and then you can flood those lines...)
        history_append(HistoryItem())
        fill_color = surface.map_rgb(self.color)
        surf_array = pygame.surfarray.pixels2d(surface)  # Create an array from the surface.
        flood_fill(surf_array, (x,y), fill_color, in_place=True)
        pygame.surfarray.blit_array(surface, surf_array)
        
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
            self.drawing_area().draw()
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
            movie.frames[movie.pos].save()
            return

        x, y = pygame.mouse.get_pos()
        for elem in self.elems:
            left, bottom, width, height = elem.rect
            if x>=left and x<left+width and y>=bottom and y<bottom+height:
                if not self.is_playing or isinstance(elem, TogglePlaybackButton):
                    self._dispatch_event(elem, event, x, y)

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
            
def pen2mask(image, rgb, transparency):
    image_alpha = pygame.Surface((image.get_width(), image.get_height()), pygame.SRCALPHA, image.copy())
    arr = pygame.surfarray.array2d(image)
    pen = image.map_rgb(PEN)
    color = image_alpha.map_rgb(rgb)
    bitmask = (arr==pen)
    arr = bitmask*color
    pygame.surfarray.blit_array(image_alpha, arr)
    image_alpha.set_alpha(int(transparency*255))
    return image_alpha

class DrawingArea:
    def __init__(self):
        pass
    def draw(self):
        left, bottom, width, height = self.rect
        frame = to_scale(movie.frames[layout.playing_index].surface if layout.is_playing else movie.curr_frame())
        screen.blit(frame, (left, bottom), (0, 0, width, height))

        if not layout.is_playing:
            mask = layout.timeline_area().combined_light_table_mask()
            if mask:
                screen.blit(mask, (left, bottom), (0, 0, width, height))

    def fix_xy(self,x,y):
        left, bottom, _, _ = self.rect
        return (x-left)*SCALE, (y-bottom)*SCALE
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
        frame = make_surface(width*SCALE, height*SCALE)
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
            else:
                mode_x = x + 2
                mode = self.arrow_icon if self.loop_mode else self.loop_icon
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
                last_modified = get_last_modified([f for frameid, f, h in clip_frame_filenames(clip)])
            except:
                continue
            self.clips.append(clip)
            self.images.append(scale_image(pg.image.load(last_modified), int(screen.get_width() * 0.15)))
        self.clip_pos = 0 
    def draw(self):
        left, bottom, width, height = self.rect
        first = True
        pos = self.show_pos if self.show_pos is not None else self.clip_pos
        for image in self.images[pos:]:
            border = 1 + first*2
            if first and pos == self.clip_pos:
                try:
                    image = scale_image(movie.curr_frame(), image.get_width()) 
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

def to_scale(frame):
    if SCALE==1:
        return frame
    _,_,w,h = frame.get_rect()
    return scale_image(frame,w/SCALE,h/SCALE)

class Frame:
    def __init__(self, surface, dir):
        self.surface = surface
        self.dir = dir
        self.id = str(uuid.uuid1())
        self.hold = False
        # we don't aim to maintain a "perfect" dirty flag such as "doing 5 things and undoing
        # them should result in dirty==False." The goal is to avoid gratuitous saving when
        # scrolling thru the timeline, which slows things down and prevents reopening
        # clips at the last actually-edited frame after exiting the program
        self.dirty = True

    def filename(self): return os.path.join(self.dir, f'{self.id}.bmp')
    def save(self):
        if self.dirty:
            pygame.image.save(self.surface, self.filename())
            self.dirty = False
    def delete(self):
        fname = self.filename()
        if os.path.exists(fname):
            os.unlink(fname)

def clip_frame_filenames(clipdir):
    with open(os.path.join(clipdir, FRAME_ORDER_FILE), 'r') as frame_order:
        frames = json.loads(frame_order.read())
        return [(frame['id'], os.path.join(clipdir, frame['id']+'.bmp'), frame['hold']) for frame in frames]

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
                frame = Frame(pg.image.load(fname).convert(), self.dir)
                frame.id = frameid
                frame.hold = hold
                frame.dirty = False
                self.frames.append(frame)
            last_modified_id = fname2id[get_last_modified(fname2id.keys())]
            # reopen at the last modified frame
            self.pos = [frame.id for frame in self.frames].index(last_modified_id)
        self.mask_cache = {}
        self.thumbnail_cache = {}

    def save_meta(self):
        frames = [{'id':frame.id,'hold':frame.hold} for frame in self.frames]
        text = json.dumps(frames,indent=2)
        with open(os.path.join(self.dir, FRAME_ORDER_FILE), 'w') as frame_order:
            frame_order.write(text)

    def get_mask(self, pos, color, transparency):
        assert pos != self.pos
        mask = self.mask_cache.setdefault(pos, LightTableMask())
        if mask.color == color and mask.transparency == transparency \
            and mask.movie_pos == self.pos and mask.movie_len == len(self.frames):
            return mask.surface
        mask.surface = to_scale(pen2mask(self.frames[pos].surface, color, transparency))
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
        thumbnail.surface = scale_image(self.frames[pos].surface, width, height)
        return thumbnail.surface

    def clear_cache(self):
        self.mask_cache = {}
        self.thumbnail_cache = {}

    def seek_frame(self,pos):
        assert pos >= 0 and pos < len(self.frames)
        self.frames[self.pos].save()
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
        self.frames[self.pos].save()
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
            self.frames[self.pos].save()
        self.pos = at_pos

        removed = self.frames[self.pos]

        del self.frames[self.pos]
        removed.delete()

        if self.pos >= len(self.frames):
            self.pos = len(self.frames)-1

        if new_pos >= 0:
            self.pos = new_pos

        self.save_meta()

        return removed

    def curr_frame(self):
        return self.frames[self.pos].surface

    def edit_curr_frame(self):
        self.frames[self.pos].dirty = True
        return self.curr_frame()

    def save_gif(self):
        with imageio.get_writer(self.dir + '.gif', fps=FRAME_RATE, mode='I') as writer:
            for frame in self.frames:
                writer.append_data(np.transpose(pygame.surfarray.pixels3d(frame.surface), [1,0,2]))

    def save_before_closing(self):
        self.frames[self.pos].dirty = True # updates the image timestamp so we open at that image next time...
        self.frames[self.pos].save()
        self.save_gif()

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
    def __init__(self, pos, frame):
        self.pos = pos
        self.frame = frame
    def undo(self):
        movie.insert_frame_at_pos(self.pos, self.frame)
    def __str__(self):
        return f'RemoveFrameHistoryItem(inserting at pos {self.pos})'

def append_seek_frame_history_item_if_frame_is_dirty():
    if history and not isinstance(history[-1], SeekFrameHistoryItem):
        history_append(SeekFrameHistoryItem(movie.pos))

def insert_frame():
    movie.insert_frame()
    history_append(InsertFrameHistoryItem(movie.pos))

def remove_frame():
    pos = movie.pos
    removed = movie.remove_frame()
    history_append(RemoveFrameHistoryItem(pos, removed))

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

def init_layout():
    screen.fill(UNDRAWABLE)

    global layout
    layout = Layout()
    layout.add((0.15,0.15,0.7,0.85), DrawingArea())
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

movie = Movie(default_clip_dir())

init_layout()

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
