import pygame
pg = pygame

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

PEN = (0 , 0 , 0)
WIDTH = 5
CURSOR_SIZE = 150

screen.fill((255,255,255))
pygame.display.flip()

def drawCircle( screen, x, y ):
  pygame.draw.circle( screen, PEN, ( x, y ), WIDTH/2 )

def drawLine(screen, pos1, pos2):
  pygame.draw.line( screen, PEN, pos1, pos2, WIDTH )

def load_cursor(file):
  surface = pg.image.load(file)
  surface = pg.transform.scale(surface, (CURSOR_SIZE, CURSOR_SIZE))
  for y in range(CURSOR_SIZE):
      for x in range(CURSOR_SIZE):
          r,g,b,a = surface.get_at((x,y))
          surface.set_at((x,y), (r,g,b,min(a,192)))
  return pg.cursors.Cursor((10,CURSOR_SIZE-7), surface)

pencil_cursor = load_cursor('pencil.png')
pg.mouse.set_cursor(pencil_cursor)

class HistoryItem:
    def __init__(self, surface):
        self.surface = surface
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
        elif event.key == ord(' '):
            if history:
                history[-1].undo()
                history.pop()
            
    if event.type == pygame.MOUSEBUTTONDOWN:
      history.append(HistoryItem(screen.copy()))
      isPressed = True
    elif event.type == pygame.MOUSEBUTTONUP:
      isPressed = False
      if prevDrawn:
          drawCircle(screen, *prevDrawn)
      if history:
          history[-1].optimize()
      pygame.display.flip()
      prevDrawn = None
    elif event.type == pygame.MOUSEMOTION and isPressed == True:
      ( x, y ) = pygame.mouse.get_pos()       # returns the position of mouse cursor
      if history:
          history[-1].affected(x-WIDTH,y-WIDTH,x+WIDTH,y+WIDTH)
      if prevDrawn:
          drawLine(screen, prevDrawn, (x,y))
      else:
          drawCircle( screen, x, y )
      prevDrawn = (x,y)
      pygame.display.flip()
   except:
    print('INTERNAL ERROR (printing and continuing)')
    import traceback
    traceback.print_exc()
      
pygame.display.quit()
pygame.quit()
