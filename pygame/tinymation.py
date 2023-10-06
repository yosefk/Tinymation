import pygame

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

BLUE = (0 , 0 , 255)

def drawCircle( screen, x, y ):
  pygame.draw.circle( screen, BLUE, ( x, y ), 5 )

isPressed = False
escape = False
while not escape: 
  for event in pygame.event.get():
    if event.type == pygame.KEYDOWN and event.key == 27: # ESC pressed
        escape = True
    if event.type == pygame.MOUSEBUTTONDOWN:
      isPressed = True
    elif event.type == pygame.MOUSEBUTTONUP:
      isPressed = False
    elif event.type == pygame.MOUSEMOTION and isPressed == True:
      ( x, y ) = pygame.mouse.get_pos()       # returns the position of mouse cursor
      drawCircle( screen, x, y )
      pygame.display.flip()
      
pygame.display.quit()
pygame.quit()
