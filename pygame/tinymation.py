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

def load_cursor(file, flip=False):
  surface = pg.image.load(file)
  surface = pg.transform.scale(surface, (CURSOR_SIZE, CURSOR_SIZE))
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
        self.surface = screen.copy()
        self.minx = 10**9
        self.miny = 10**9
        self.maxx = -10**9
        self.maxy = -10**9
        self.optimized = False
    def undo(self):
        if self.optimized:
            screen.blit(self.surface, (self.minx, self.miny), (0, 0, self.maxx-self.minx+1, self.maxy-self.miny+1))
        else:
            screen.blit(self.surface, screen.get_rect())
        pygame.display.flip()
    def affected(self,minx,miny,maxx,maxy):
        self.minx = min(minx,self.minx)
        self.maxx = max(maxx,self.maxx)
        self.miny = min(miny,self.miny)
        self.maxy = max(maxy,self.maxy)
    def optimize(self):
        if self.minx == 10**9:
            return
        left, bottom, width,height = screen.get_rect()
        right = left+width
        top = bottom+height
        self.minx = max(self.minx, left)
        self.maxx = min(self.maxx, right-1)
        self.miny = max(self.miny, bottom)
        self.maxy = min(self.maxy, top-1)
        
        affected = pg.Surface((self.maxx-self.minx+1, self.maxy-self.miny+1), screen.get_flags(), screen.get_bitsize(), screen.get_masks())
        affected.blit(self.surface, (0,0), (self.minx, self.miny, self.maxx+1, self.maxy+1))
        self.surface = affected
        self.optimized = True

class PenTool:
    def __init__(self, color=PEN, width=WIDTH):
        self.prev_drawn = None
        self.color = color
        self.width = width

    def draw(self, rect, cursor_surface):
        left, bottom, width, height = rect
        surface = pg.transform.scale(cursor_surface, (width, height))
        print(width, height)
        screen.blit(surface, (left, bottom), (0, 0, width, height))

    def on_mouse_down(self, x, y):
        history.append(HistoryItem())

    def on_mouse_up(self, x, y):
        self.prev_drawn = None
        if history:
            history[-1].optimize()

    def on_mouse_move(self, x, y):
       if history:
            history[-1].affected(x-self.width,y-self.width,x+self.width,y+self.width)
       if self.prev_drawn:
            drawLine(screen, self.prev_drawn, (x,y), self.color, self.width)
       drawCircle( screen, x, y, self.color, self.width )
       self.prev_drawn = (x,y)

class PaintBucketTool:
    def __init__(self,color):
        self.color = color
    def draw(self, rect, cursor_surface):
        pygame.draw.rect(screen, self.color, rect)
    def on_mouse_down(self, x, y):
        history.append(HistoryItem())
        # TODO: would be better to optimize the history item
        surface = screen
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
        self.tool = PenTool()

    def add(self, rect, elem):
        left, bottom, width, height = rect
        srect = (int(left*self.width), int(bottom*self.height), int(width*self.width), int(height*self.height))
        self.elems.append((srect, elem))
        elem.draw(srect)
        pygame.draw.rect(screen, PEN, srect, 1, 1)

    def on_event(self,event):
        x, y = pygame.mouse.get_pos()
        for rect, elem in self.elems:
            left, bottom, width, height = rect
            if x>=left and x<left+width and y>=bottom and y<bottom+height:
                # mouse position is within this element
                self._dispatch_event(elem, event, x, y)

    def _dispatch_event(self, elem, event, x, y):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.is_pressed = True
            elem.on_mouse_down(x,y)
            pygame.display.flip()
        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_pressed = False
            elem.on_mouse_up(x,y)
            pygame.display.flip()
        elif event.type == pygame.MOUSEMOTION and self.is_pressed:
            elem.on_mouse_move(x,y)
            pygame.display.flip()
            
class DrawingArea:
    def __init__(self):
        pass
    def draw(self,rect): pass
    def on_mouse_down(self,x,y):
        layout.tool.on_mouse_down(x,y)
    def on_mouse_up(self,x,y):
        layout.tool.on_mouse_up(x,y)
    def on_mouse_move(self,x,y):
        layout.tool.on_mouse_move(x,y)

class ToolSelectionButton:
    def __init__(self, tool):
        self.tool = tool
    def draw(self,rect):
        self.tool.tool.draw(rect,self.tool.cursor[1])
    def on_mouse_down(self,x,y):
        set_tool(self.tool)
    def on_mouse_up(self,x,y): pass
    def on_mouse_move(self,x,y): pass

Tool = collections.namedtuple('Tool', ['tool', 'cursor', 'chars'])

TOOLS = {
    'pencil': Tool(PenTool(), pencil_cursor, 'bB'),
    'eraser': Tool(PenTool(BACKGROUND, WIDTH), eraser_cursor, 'eE'),
    'eraser-medium': Tool(PenTool(BACKGROUND, WIDTH*5), eraser_cursor, 'rR'),
    'eraser-big': Tool(PenTool(BACKGROUND, WIDTH*20), eraser_cursor, 'tT'),
}

def set_tool(tool):
    layout.tool = tool.tool
    pg.mouse.set_cursor(tool.cursor[0])

def init_layout():
    screen.fill(BACKGROUND)

    global layout
    layout = Layout()
    layout.add((0.15,0.15,0.85,0.85), DrawingArea())
    layout.add((0,0.85,0.075, 0.15), ToolSelectionButton(TOOLS['pencil']))
    color_w = 0.025
    i = 0
    
    for y in np.arange(0.3,0.85-0.001,color_w):
        for x in np.arange(0,0.15-0.001,color_w):            
            rgb = pygame.Color(0)
            rgb.hsla = (i*10 % 360, 50, 50, 100)
            tool = Tool(PaintBucketTool(rgb), eraser_cursor, '')
            layout.add((x,y,color_w,color_w), ToolSelectionButton(tool))
            i += 1

    pygame.display.flip()

init_layout()

history = []
escape = False
while not escape: 
  for event in pygame.event.get():
   try:
      if event.type == pygame.KEYDOWN:
        if event.key == 27: # ESC pressed
            escape = True
  
        if layout.is_pressed:
            continue # ignore keystrokes (except ESC) when a mouse tool is being used
        
        if event.key == ord(' '): # undo
            if history:
                history[-1].undo()
                history.pop()

        for tool in TOOLS.values():
            if event.key in [ord(c) for c in tool.chars]:
                set_tool(tool)
                
      else:
          layout.on_event(event)
   except:
    print('INTERNAL ERROR (printing and continuing)')
    import traceback
    traceback.print_exc()
      
pygame.display.quit()
pygame.quit()
