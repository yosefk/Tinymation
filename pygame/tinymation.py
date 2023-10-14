import pygame
import collections
import numpy as np

# this requires numpy to be installed in addition to scikit-image
from skimage.morphology import flood_fill
pg = pygame

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

PEN = (20, 20, 20)
BACKGROUND = (240, 235, 220)
WIDTH = 5
CURSOR_SIZE = 150

def drawCircle( screen, x, y, color, width):
  pygame.draw.circle( screen, color, ( x, y ), width/2 )

def drawLine(screen, pos1, pos2, color, width):
  pygame.draw.line( screen, color, pos1, pos2, width )

def make_surface(width, height):
    return pg.Surface((width, height), screen.get_flags(), screen.get_bitsize(), screen.get_masks())

def scale_image(surface, width, height):
    return pg.transform.smoothscale(surface, (width, height))

def load_cursor(file, flip=False):
  surface = pg.image.load(file)
  surface = scale_image(surface, CURSOR_SIZE, CURSOR_SIZE)#pg.transform.scale(surface, (CURSOR_SIZE, CURSOR_SIZE))
  if flip:
      surface = pg.transform.flip(surface, True, True)
  for y in range(CURSOR_SIZE):
      for x in range(CURSOR_SIZE):
          r,g,b,a = surface.get_at((x,y))
          surface.set_at((x,y), (r,g,b,min(a,192)))
  return pg.cursors.Cursor((10,CURSOR_SIZE-7), surface), surface

pencil_cursor = load_cursor('pencil.png')
eraser_cursor = load_cursor('pencil.png',flip=True)
pg.mouse.set_cursor(pencil_cursor[0])

class HistoryItem:
    def __init__(self):
        self.surface = movie.curr_frame().copy()
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
        frame = movie.curr_frame()
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
        left, bottom, width,height = movie.curr_frame().get_rect()
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

class PenTool:
    def __init__(self, color=PEN, width=WIDTH):
        self.prev_drawn = None
        self.color = color
        self.width = width
        self.history_item = None

    def draw(self, rect, cursor_surface):
        left, bottom, width, height = rect
        surface = scale_image(cursor_surface, width, height)
        screen.blit(surface, (left, bottom), (0, 0, width, height))

    def on_mouse_down(self, x, y):
        self.history_item = HistoryItem()

    def on_mouse_up(self, x, y):
        self.prev_drawn = None
        if self.history_item:
            self.history_item.optimize()
            history.append(self.history_item)
            self.history_item = None

    def on_mouse_move(self, x, y):
       if self.history_item:
            self.history_item.affected(x-self.width,y-self.width,x+self.width,y+self.width)
       frame = movie.curr_frame()
       if self.prev_drawn:
            drawLine(frame, self.prev_drawn, (x,y), self.color, self.width)
       drawCircle(frame, x, y, self.color, self.width)
       self.prev_drawn = (x,y)

class PaintBucketTool:
    def __init__(self,color):
        self.color = color
    def draw(self, rect, cursor_surface):
        pygame.draw.rect(screen, self.color, rect)
    def on_mouse_down(self, x, y):
        history.append(HistoryItem())
        # TODO: would be better to optimize the history item
        surface = movie.curr_frame()
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

    def add(self, rect, elem):
        left, bottom, width, height = rect
        srect = (int(left*self.width), int(bottom*self.height), int(width*self.width), int(height*self.height))
        elem.rect = srect
        self.elems.append(elem)

    def draw(self):
        if self.is_pressed and self.focus_elem is self.drawing_area():
            self.drawing_area().draw()
            return
        screen.fill(BACKGROUND)
        for elem in self.elems:
            elem.draw()
            pygame.draw.rect(screen, PEN, elem.rect, 1, 1)

    def on_event(self,event):
        if self.is_playing:
            # TODO: this isn't the way - should allow the stop button to be pressed
            # need to disable the other operations differently from this
            if event.type == TIMER_EVENT:
                self.playing_index = (self.playing_index + 1) % len(movie.frames)
            return
        
        x, y = pygame.mouse.get_pos()
        for elem in self.elems:
            left, bottom, width, height = elem.rect
            if x>=left and x<left+width and y>=bottom and y<bottom+height:
                self.focus_elem = elem
                # mouse position is within this element
                self._dispatch_event(elem, event, x, y)

    def _dispatch_event(self, elem, event, x, y):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.is_pressed = True
            elem.on_mouse_down(x,y)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_pressed = False
            elem.on_mouse_up(x,y)
        elif event.type == pygame.MOUSEMOTION and self.is_pressed:
            elem.on_mouse_move(x,y)

    def drawing_area(self):
        assert isinstance(self.elems[0], DrawingArea)
        return self.elems[0]

    def toggle_playing(self):
        self.is_playing = not self.is_playing
            
class DrawingArea:
    def __init__(self):
        pass
    def draw(self):
        try:
            m = movie
        except:
            return
        left, bottom, width, height = self.rect
        frame = m.frames[layout.playing_index] if layout.is_playing else m.curr_frame()
        screen.blit(frame, (left, bottom), (0, 0, width, height))
    def on_mouse_down(self,x,y):
        left, bottom, _, _ = self.rect
        layout.tool.on_mouse_down(x-left,y-bottom)
    def on_mouse_up(self,x,y):
        left, bottom, _, _ = self.rect
        layout.tool.on_mouse_up(x-left,y-bottom)
    def on_mouse_move(self,x,y):
        left, bottom, _, _ = self.rect
        layout.tool.on_mouse_move(x-left,y-bottom)
    def get_frame(self):
        _, _, width, height = self.rect
        frame = make_surface(width, height)
        frame.blit(screen, (0,0), self.rect)
        return frame
    def new_frame(self):
        _, _, width, height = self.rect
        frame = make_surface(width, height)
        frame.fill(BACKGROUND)
        return frame

class TimelineArea:
    def __init__(self):
        self.frame_boundaries = set()
        self.prevx = 0
    def x2frame(self, x):
        for left, right, pos in self.frame_boundaries:
            if x >= left and x <= right:
                return pos
    def draw(self):
        try:
            m = movie
        except:
            return
        left, bottom, width, height = self.rect
        frame_width = movie.curr_frame().get_width()
        frame_height = movie.curr_frame().get_height()
        #thumb_width = movie.curr_frame().get_width() * height // movie.curr_frame().get_height()
        x = left
        factors = [0.7,0.6,0.5,0.4,0.3,0.2,0.14]
        i = 0

        self.frame_boundaries = set()

        def draw_frame(frame, x, thumb_width):
            surface, pos = frame
            scaled = scale_image(surface, thumb_width, height)
            screen.blit(scaled, (x, bottom), (0, 0, thumb_width, height))
            pygame.draw.rect(screen, PEN, (x, bottom, thumb_width, height), 1, 1)
            self.frame_boundaries.add((x, x+thumb_width, pos))

        def thumb_width(factor):
            return int((frame_width * height // frame_height) * factor)

        # current frame
        curr_frame_width = thumb_width(1)
        centerx = (left+width)/2
        draw_frame((movie.curr_frame(), movie.pos), centerx - curr_frame_width/2, curr_frame_width)

        # next frames
        x = centerx + curr_frame_width/2
        for i,frame in enumerate(movie.frames_and_positions(movie.pos+1,len(movie.frames))):
            if i >= len(factors):
                break
            ith_frame_width = thumb_width(factors[i])
            draw_frame(frame, x, ith_frame_width)
            x += ith_frame_width

        # previous frames
        x = centerx - curr_frame_width/2
        for i,frame in enumerate(reversed(list(movie.frames_and_positions(0,movie.pos)))):
            if i >= len(factors):
                break
            ith_frame_width = thumb_width(factors[i])
            x -= ith_frame_width
            draw_frame(frame, x, ith_frame_width)

    def on_mouse_down(self,x,y):
        print('down')
        self.prevx = x
    def on_mouse_up(self,x,y):
        self.on_mouse_move(x,y)
        print('up')
    def on_mouse_move(self,x,y):
        print('should diff',x,self.prevx)
        prev_pos = self.x2frame(self.prevx)
        curr_pos = self.x2frame(x)
        print(prev_pos, curr_pos)
        if prev_pos is None and curr_pos is None:
            self.prevx = x
            print('ret')
            return
        if curr_pos is not None and prev_pos is not None:
            pos_dist = prev_pos - curr_pos
        else:
            print('checking x vs prevx',x,self.prevx,x>self.prevx)
            pos_dist = -1 if x > self.prevx else 1
        self.prevx = x
        print(pos_dist)
        if pos_dist != 0:
            movie.seek_frame(min(max(0, movie.pos + pos_dist), len(movie.frames)-1))

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
    def draw(self):
        # TODO: show it was pressed (tool selection button shows it by changing the cursor, maybe still should show it was pressed)
        if self.icon:
            left, bottom, width, height = rect
            surface = pg.transform.scale(cursor_surface, (width, height))
            screen.blit(surface, (left, bottom), (0, 0, width, height))
    def on_mouse_down(self,x,y):
        self.function()
    def on_mouse_up(self,x,y): pass
    def on_mouse_move(self,x,y): pass

Tool = collections.namedtuple('Tool', ['tool', 'cursor', 'chars'])

TOOLS = {
    'pencil': Tool(PenTool(), pencil_cursor, 'bB'),
    'eraser': Tool(PenTool(BACKGROUND, WIDTH), eraser_cursor, 'eE'),
    'eraser-medium': Tool(PenTool(BACKGROUND, WIDTH*5), eraser_cursor, 'rR'),
    'eraser-big': Tool(PenTool(BACKGROUND, WIDTH*20), eraser_cursor, 'tT'),
}

class Movie:
    def __init__(self):
        self.frames = [layout.drawing_area().get_frame()]
        self.thumbnails = []
        self.pos = 0

    def seek_frame(self,pos):
        assert pos >= 0 and pos < len(self.frames)
        self.pos = pos

    def next_frame(self): self.seek_frame((self.pos + 1) % len(self.frames))
    def prev_frame(self): self.seek_frame((self.pos - 1) % len(self.frames))

    def insert_frame(self):
        self.frames.insert(self.pos+1, layout.drawing_area().new_frame())
        self.next_frame()

    def insert_frame_at_pos(self, pos, frame):
        assert pos >= 0 and pos <= len(self.frames)
        self.pos = pos
        self.frames.insert(self.pos, frame)

    # TODO: this works with pos modified from the outside but it's scary as the API
    def remove_frame(self, new_pos=-1):
        if len(self.frames) <= 1:
            return

        removed = self.frames[self.pos]
        del self.frames[self.pos]
        if self.pos >= len(self.frames):
            self.pos = 0

        if new_pos >= 0:
            self.pos = new_pos

        return removed

    def curr_frame(self):
        return self.frames[self.pos]

    def frames_and_positions(self, pos_begin, pos_end):
        return zip(self.frames[pos_begin:pos_end], range(pos_begin, pos_end))

class SeekFrameHistoryItem:
    def __init__(self, pos): self.pos = pos
    def undo(self): movie.seek_frame(self.pos) 
    def __str__(self): return f'SeekFrameHistoryItem(restoring pos to {self.pos})'

class InsertFrameHistoryItem:
    def __init__(self, pos): self.pos = pos
    def undo(self):
        movie.pos = self.pos
        # normally remove_frame brings you to the next frame after the one you removed.
        # but when undoing insert_frame, we bring you to the previous frame after the one
        # you removed - it's the one where you inserted the frame we're now removing to undo
        # the insert, so this is where we should go to bring you back in time.
        movie.remove_frame(new_pos=(self.pos-1)%len(movie.frames))
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
        history.append(SeekFrameHistoryItem(movie.pos))

def insert_frame():
    movie.insert_frame()
    history.append(InsertFrameHistoryItem(movie.pos))

def remove_frame():
    pos = movie.pos
    removed = movie.remove_frame()
    history.append(RemoveFrameHistoryItem(pos, removed))

def next_frame():
    if movie.pos >= len(movie.frames)-1:
        return
    append_seek_frame_history_item_if_frame_is_dirty()
    movie.next_frame()

def prev_frame():
    if movie.pos <= 0:
        return
    append_seek_frame_history_item_if_frame_is_dirty()
    movie.prev_frame()

def toggle_playing(): layout.toggle_playing()

FUNCTIONS = {
    'insert-frame': (insert_frame, '=+'),
    'remove-frame': (remove_frame, '-_'),
    'next-frame': (next_frame, '.<'),
    'prev-frame': (prev_frame, ',>'),
    'toggle-playing': (toggle_playing, '\r'),
}

def set_tool(tool):
    layout.tool = tool.tool
    if tool.cursor:
        pg.mouse.set_cursor(tool.cursor[0])

def init_layout():
    screen.fill(BACKGROUND)

    global layout
    layout = Layout()
    layout.add((0.15,0.15,0.85,0.85), DrawingArea())
    layout.add((0, 0, 1, 0.15), TimelineArea())
    layout.add((0,0.85,0.075, 0.15), ToolSelectionButton(TOOLS['pencil']))
    layout.add((0,0.15,0.075, 0.3), FunctionButton(insert_frame))
    color_w = 0.025
    i = 0
    
    for y in np.arange(0.3,0.85-0.001,color_w):
        for x in np.arange(0,0.15-0.001,color_w):            
            rgb = pygame.Color(0)
            rgb.hsla = (i*10 % 360, 50, 50, 100)
            tool = Tool(PaintBucketTool(rgb), eraser_cursor, '')
            layout.add((x,y,color_w,color_w), ToolSelectionButton(tool))
            i += 1

    layout.draw()

init_layout()

movie = Movie()

# The history is "global" for all operations. In some (rare) animation programs
# there's a history per frame. One problem with this is how to undo timeline
# operations like frame deletions (do you have a separate undo function for this?)
# It's also somewhat less intuitive in that you might have long forgotten
# what you've done on some frame when you visit it and press undo one time
# too many
history = []
escape = False

TIMER_EVENT = pygame.USEREVENT + 1

time_delay = 1000//12 # we play at 12 fps
timer_event = TIMER_EVENT
pygame.time.set_timer(timer_event, time_delay)

interesting_events = [
    pygame.KEYDOWN,
    pygame.MOUSEMOTION,
    pygame.MOUSEBUTTONDOWN,
    pygame.MOUSEBUTTONUP,
    TIMER_EVENT,
]

while not escape: 
  for event in pygame.event.get():
   if event.type not in interesting_events:
       continue
   try:
      if event.type == pygame.KEYDOWN:
        if event.key == 27: # ESC pressed
            escape = True

        if layout.is_pressed:
            continue # ignore keystrokes (except ESC) when a mouse tool is being used
        
        if event.key == ord(' '): # undo
            if history:
                # TODO: we might want a loop here since some undo ops
                # turn out to be "no-ops" (specifically seek frame where we're already there.)
                # as it is right now, you might press undo and nothing will happen which might be confusing
                history[-1].undo()
                history.pop()

        for tool in TOOLS.values():
            if event.key in [ord(c) for c in tool.chars]:
                set_tool(tool)

        for func, chars in FUNCTIONS.values():
            if event.key in [ord(c) for c in chars]:
                func()
                
      else:
          layout.on_event(event)

      # TODO: might be good to optimize repainting beyond "just repaint everything
      # upon every event"
      layout.draw()
      pygame.display.flip()
   except:
    print('INTERNAL ERROR (printing and continuing)')
    import traceback
    traceback.print_exc()
      
pygame.display.quit()
pygame.quit()
