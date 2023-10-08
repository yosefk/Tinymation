import pygame
# this requires numpy to be installed in addition to scikit-image
from skimage.morphology import flood_fill
pg = pygame

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

PEN = (20, 20, 20)
BACKGROUND = (240, 235, 220)
WIDTH = 5
CURSOR_SIZE = 150

screen.fill(BACKGROUND)
pygame.display.flip()

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
  return pg.cursors.Cursor((10,CURSOR_SIZE-7), surface)

pencil_cursor = load_cursor('pencil.png')
eraser_cursor = load_cursor('pencil.png',flip=True)
pg.mouse.set_cursor(pencil_cursor)

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
        left, bottom, right,top = screen.get_rect()
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
    def __init__(self):
        pass
    def on_mouse_down(self, x, y):
        history.append(HistoryItem())
        # TODO: would be better to optimize the history item
        surface = screen
        fill_color = surface.map_rgb((240, 240, 20)) # TODO: make configurable
        surf_array = pygame.surfarray.pixels2d(surface)  # Create an array from the surface.
        flood_fill(surf_array, (x,y), fill_color, in_place=True)
        pygame.surfarray.blit_array(surface, surf_array)
        
    def on_mouse_up(self, x, y):
        pass
    def on_mouse_move(self, x, y):
        pass

tool = PenTool()
history = []
isPressed = False
escape = False
prevDrawn = None
while not escape: 
  for event in pygame.event.get():
   try:
    if event.type == pygame.KEYDOWN:
        if event.key == 27: # ESC pressed
            escape = True
            
        if isPressed:
            continue # ignore keystrokes (except ESC) when a mouse tool is being used
        
        if event.key == ord(' '): # undo
            if history:
                history[-1].undo()
                history.pop()

        # TODO: condition on "adult mode"
        elif event.key in [ord('e'), ord('E')]:
            tool = PenTool(BACKGROUND, WIDTH)
            pg.mouse.set_cursor(eraser_cursor)
        elif event.key in [ord('r'), ord('R')]:
            tool = PenTool(BACKGROUND, WIDTH*5)
        elif event.key in [ord('t'), ord('T')]:
            tool = PenTool(BACKGROUND, WIDTH*20)
        elif event.key in [ord('b'), ord('B')]:
            tool = PenTool()
            pg.mouse.set_cursor(pencil_cursor)
        elif event.key in [ord('k'), ord('K')]:
            tool = PaintBucketTool()
            
    if event.type == pygame.MOUSEBUTTONDOWN:
      isPressed = True
      tool.on_mouse_down(*pygame.mouse.get_pos())
      pygame.display.flip()
    elif event.type == pygame.MOUSEBUTTONUP:
      isPressed = False
      tool.on_mouse_up(*pygame.mouse.get_pos())
      pygame.display.flip()
    elif event.type == pygame.MOUSEMOTION and isPressed == True:
      tool.on_mouse_move(*pygame.mouse.get_pos())
      pygame.display.flip()
   except:
    print('INTERNAL ERROR (printing and continuing)')
    import traceback
    traceback.print_exc()
      
pygame.display.quit()
pygame.quit()
