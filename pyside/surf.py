import pygame as pg
import pygame.gfxdraw

SRCALPHA = pg.SRCALPHA

class Surface:
    def __init__(self, first, *rest):
        if isinstance(first, Surface):
            assert len(rest) == 0
            self.surface = first.surface
        elif isinstance(first, pg.Surface):
            assert len(rest) == 0
            self.surface = first
        else:
            self.surface = pg.Surface(first, *rest)

    def get_width(self):
        return self.surface.get_width()

    def get_height(self):
        return self.surface.get_height()

    def get_rect(self, **kwargs):
        return self.surface.get_rect(**kwargs)

    def fill(self, color, rect=None, special_flags=0):
        return self.surface.fill(color, rect, special_flags)

    def blit(self, source, dest, area=None, special_flags=0):
        if isinstance(source, Surface):
            source = source.surface
        return self.surface.blit(source, dest, area, special_flags)

    def blits(self, blit_sequence):
        new_seq = [(item[0].surface,)+item[1:] for item in blit_sequence]
        return self.surface.blits(new_seq)

    def copy(self):
        return Surface(self.surface.copy())

    def subsurface(self, *args):
        return Surface(self.surface.subsurface(*args))

    def set_alpha(self, value, flags=0):
        return self.surface.set_alpha(value, flags)

    def get_alpha(self):
        return self.surface.get_alpha()

    def get_at(self, pos):
        return self.surface.get_at(pos)

    def set_clip(self, rect):
        return self.surface.set_clip(rect)

def load(fname):
    return Surface(pg.image.load(fname))

def save(surface, filename):
    return pg.image.save(surface.surface, filename)

def rotate(surface, angle):
    return Surface(pg.transform.rotate(surface.surface, angle))

def smoothscale(surface, size):
    return Surface(pg.transform.smoothscale(surface.surface, size))

def pixels3d(surface):
    return pg.surfarray.pixels3d(surface.surface)

def pixels_alpha(surface):
    return pg.surfarray.pixels_alpha(surface.surface)

def box(surface, rect, color):
    return pg.gfxdraw.box(surface.surface, rect, color)

def rect(surface, color, rect, *rest):
    return pg.draw.rect(surface.surface, color, rect, *rest)

def filled_circle(surface, x, y, radius, color):
    return pg.gfxdraw.filled_circle(surface.surface, x, y, radius, color)
