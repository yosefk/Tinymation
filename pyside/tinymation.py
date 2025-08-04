import numpy as np
import sys
import os
from PySide6.QtGui import QImage

on_windows = os.name == 'nt'
on_linux = sys.platform == 'linux'

ASSETS = 'assets'

# for the saving timer
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=1)

import signal
import uuid
import json

import res

import surf
from surf import Surface

FRAME_RATE = 12
CLIP_FILE = 'movie.json' # on Windows, this starting with 'm' while frame0000.png starts with 'f'
# makes the png the image inside the directory icon displayed in Explorer... which is nice
FRAME_FMT = 'frame%04d.png'
CURRENT_FRAME_FILE = 'current_frame.png'
PALETTE_FILE = 'palette.png'
BACKGROUND = (240, 235, 220)
PEN = (20, 20, 20)

from cache import Cache, CachedItem
cache = Cache()

def fit_to_resolution(surface):
    w,h = surface.get_width(), surface.get_height()
    if w == res.IWIDTH and h == res.IHEIGHT:
        return surface
    elif w == res.IHEIGHT and h == res.IWIDTH:
        return surf.rotate(surface, 90 * (1 if w>h else -1)) 
    else:
        assert False, f'only supporting {res.IWIDTH}x{res.IHEIGHT} or {res.IHEIGHT}x{res.IWIDTH} images, got {w}x{h}'

def new_frame():
    return Surface((res.IWIDTH, res.IHEIGHT), color=BACKGROUND + (0,))

def load_image(fname):
    if not os.path.dirname(fname) and not os.path.exists(fname):
        asset = os.path.join(ASSETS, fname)
        if os.path.exists(asset):
            fname=asset
    return surf.load(fname)

class Frame:
    def __init__(self, dir, layer_id=None, frame_id=None, read_pixels=True):
        self.dir = dir
        self.layer_id = layer_id
        if frame_id is not None: # id - load the surfaces from the directory
            self.id = frame_id
            self.del_pixels()
            if read_pixels:
                self.read_pixels()
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

        self.compression_future = None

    def mark_as_garbage_in_cache(self):
        cache.delete_id(self.cache_id())

    def read_pixels(self):
        for surf_id in self.surf_ids():
            fname = self.filename(surf_id)
            if os.path.exists(fname):
                setattr(self,surf_id,fit_to_resolution(surf.load(fname)))

    def del_pixels(self):
        for surf_id in self.surf_ids():
            setattr(self,surf_id,None)

    def empty(self): return self.color is None

    def _create_surfaces_if_needed(self):
        if not self.empty():
            return
        self.color = new_frame()
        self.lines = Surface((self.color.get_width(), self.color.get_height()), color=PEN)
        surf.pixels_alpha(self.lines)[:] = 0

    def get_content(self): return self.color.copy(), self.lines.copy()
    def set_content(self, content):
        color, lines = content
        self.color = fit_to_resolution(color.copy())
        self.lines = fit_to_resolution(lines.copy())
    def clear(self):
        self.color = None
        self.lines = None

    def increment_version(self):
        self._create_surfaces_if_needed()
        self.dirty = True
        self.version += 1
        cache.update_id(self.cache_id(), self.version)

    def surf_ids(self): return ['lines','color']
    def get_width(self): return res.IWIDTH
    def get_height(self): return res.IHEIGHT
    def get_rect(self): return empty_frame().color.get_rect()

    def surf_by_id(self, surface_id):
        s = getattr(self, surface_id)
        return s if s is not None else empty_frame().surf_by_id(surface_id)

    def surface(self, roi=None):
        def sub(surface): return surface.subsurface(roi) if roi else surface
        if self.empty():
            return sub(empty_frame().color)
        subc = sub(self.color)
        s = subc.empty_like()
        subc.blit(sub(self.lines), (0, 0), into=s)
        return s

    def thumbnail(self, width=None, height=None, roi=None, inv_scale=None):
        if self.empty():
            if inv_scale is not None:
                width = round(roi[2] * inv_scale)
                height = round(roi[3] * inv_scale)
            return large_empty_surface(width, height)

        return scale_image(self.surface(roi), width, height, inv_scale)
        # note that for a small ROI it's faster to blit lines onto color first, and then scale;
        # for a large ROI, it's faster to scale first and then blit the smaller number of pixels.
        # however this produces ugly artifacts where lines & color are eroded and you see through
        # both into the layer below, so we don't do it

    def filename(self,surface_id):
        fname = f'{self.id}-{surface_id}.'
        if self.layer_id:
            fname = os.path.join(f'layer-{self.layer_id}', fname)
        fname = os.path.join(self.dir, fname)
        return fname+'png'
    def wait_for_compression_to_finish(self):
        if self.compression_future:
            self.compression_future.result()
        self.compression_future = None
    def _save_to_files(self, fnames_and_surfaces):
        for fname, surface in fnames_and_surfaces:
            surf.save(surface, fname)
    def save(self):
        if self.dirty:
            self.wait_for_compression_to_finish()
            to_save = [(self.filename(surf_id), self.surf_by_id(surf_id).copy()) for surf_id in self.surf_ids()]
            self.compression_future = executor.submit(self._save_to_files, to_save)
            self.dirty = False
    def delete(self):
        self.wait_for_compression_to_finish()
        for surf_id in self.surf_ids():
            fname = self.filename(surf_id)
            if os.path.exists(fname):
                os.unlink(fname)

    def size(self):
        # a frame is 2 RGBA surfaces
        return (self.get_width() * self.get_height() * 8) if not self.empty() else 0

    def cache_id(self): return (self.id, self.layer_id) if not self.empty() else None
    def cache_id_version(self): return self.cache_id(), self.version

    def fit_to_resolution(self):
        if self.empty():
            return
        for surf_id in self.surf_ids():
            setattr(self, surf_id, fit_to_resolution(self.surf_by_id(surf_id)))

_empty_frame = Frame('')
def empty_frame():
    global _empty_frame
    if not _empty_frame.empty() and (_empty_frame.color.get_width() != res.IWIDTH or _empty_frame.color.get_height() != res.IHEIGHT):
        _empty_frame = Frame('')
    _empty_frame._create_surfaces_if_needed()
    return _empty_frame

_large_empty_surface = None
def large_empty_surface(width, height):
    global _large_empty_surface
    if _large_empty_surface is None or _large_empty_surface.get_width() < width or _large_empty_surface.get_height() < height:
        _large_empty_surface = Surface((width*2, height*2))

    return _large_empty_surface.subsurface(0, 0, width, height)

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

def default_progress_callback(done_items, total_items): pass

class MovieData:
    def __init__(self, dir, read_pixels=True, progress=default_progress_callback):
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

            movie_width, movie_height = clip.get('resolution',(res.IWIDTH,res.IHEIGHT))
            frame_ids = clip['frame_order']
            layer_ids = clip['layer_order']
            holds = clip['hold']
            visible = clip.get('layer_visible', [True]*len(layer_ids))
            locked = clip.get('layer_locked', [False]*len(layer_ids))

            res.set_resolution(movie_width, movie_height)

            done = 0
            total = len(layer_ids) * len(frame_ids)

            self.layers = []
            for layer_index, layer_id in enumerate(layer_ids):
                frames = []
                for frame_index, frame_id in enumerate(frame_ids):
                    frame = Frame(dir, layer_id, frame_id, read_pixels=read_pixels)
                    frame.hold = holds[layer_index][frame_index]
                    frames.append(frame)

                    done += 1
                    progress(done, total)

                layer = Layer(frames, dir, layer_id)
                layer.visible = visible[layer_index]
                layer.locked = locked[layer_index]
                self.layers.append(layer)

            self.pos = clip['frame_pos']
            self.layer_pos = clip['layer_pos']
            self.frames = self.layers[self.layer_pos].frames

            # we can't update Layout at this point since upon startup we load a clip
            # before initializing the layout [since we don't know the aspect ratio we need
            # before loading the clip...]
            self.loaded_on_light_table = dict((int(k),v) for k,v in clip.get('on_light_table', {}).items())
            self.loaded_zoom = clip.get('zoom', 1)
            self.loaded_xyoffset = clip.get('xyoffset', [0, 0])
            self.loaded_zoom_center = clip.get('zoom_center', [0, 0])

    def restore_viewing_params(self):
        if getattr(self, 'loaded_zoom', None) is None:
            return
        da = layout.drawing_area()
        da.set_zoom(self.loaded_zoom)
        da.set_xyoffset(*self.loaded_xyoffset)
        da.set_zoom_center(self.loaded_zoom_center)
        if self.loaded_on_light_table:
            layout.timeline_area().on_light_table = self.loaded_on_light_table

    def save_meta(self):
        try:
            da = layout.drawing_area()
        except:
            return
        clip = {
            'resolution':[res.IWIDTH, res.IHEIGHT],
            'frame_pos':self.pos,
            'layer_pos':self.layer_pos,
            'frame_order':[frame.id for frame in self.frames],
            'layer_order':[layer.id for layer in self.layers],
            'layer_visible':[layer.visible for layer in self.layers],
            'layer_locked':[layer.locked for layer in self.layers],
            'hold':[[frame.hold for frame in layer.frames] for layer in self.layers],
            'on_light_table':layout.timeline_area().on_light_table,
            'zoom':da.zoom,
            'xyoffset':[da.xoffset, da.yoffset],
            'zoom_center':list(da.zoom_center),
        }
        fname = os.path.join(self.dir, CLIP_FILE)
        text = json.dumps(clip,indent=2)
        try:
            with open(fname) as clip_file:
                if text == clip_file.read():
                    return # no changes
        except FileNotFoundError:
            pass
        with open(fname, 'w') as clip_file:
            clip_file.write(text)

    def gif_path(self): return os.path.realpath(self.dir)+'-GIF.gif'
    def mp4_path(self): return os.path.realpath(self.dir)+'-MP4.mp4'
    def still_png_path(self): return os.path.realpath(self.dir)+'-PNG.png'
    def export_paths_outside_clipdir(self): return [self.gif_path(), self.mp4_path(), self.still_png_path()]
    def png_path(self, i): return os.path.join(os.path.realpath(self.dir), FRAME_FMT%i)
    def png_wildcard(self): return os.path.join(os.path.realpath(self.dir), 'frame*.png')

    def exported_files_exist(self):
        if not os.path.exists(self.gif_path()) or not os.path.exists(self.mp4_path()):
            return False
        for i in range(len(self.frames)):
            if not os.path.exists(self.png_path(i)):
                return False
        return True

    def _blit_layers(self, layers, pos, transparent=False, include_invisible=False, width=None, height=None, roi=None, inv_scale=None):
        if not inv_scale:
            if not width: width=res.IWIDTH
            if not height: height=res.IHEIGHT
        if not roi: roi = (0, 0, res.IWIDTH, res.IHEIGHT)
        s = Surface((width if width else round(roi[2]*inv_scale), height if height else round(roi[3]*inv_scale)), color=None if transparent else BACKGROUND)
        surfaces = []
        for layer in layers:
            if not layer.visible and not include_invisible:
                continue
            if width==res.IWIDTH and height==res.IHEIGHT and roi==(0,0,res.IWIDTH,res.IHEIGHT):
                f = layer.frame(pos)
                surfaces.append(f.surf_by_id('color'))
                surfaces.append(f.surf_by_id('lines'))
            else:
                surfaces.append(movie.get_thumbnail(pos, width, height, transparent_single_layer=self.layers.index(layer), roi=roi, inv_scale=inv_scale))
        s.blits(surfaces)
        return s


# a supposed advantage of this verbose method of writing MP4s using PyUV over "just" using imageio
# is that `pip install av` installs ffmpeg libraries so you don't need to worry
# separately about installing ffmpeg. imageio also fails in a "TiffWriter" at the
# time of writing, or if the "fps" parameter is removed, creates a giant .mp4
# output file that nothing seems to be able to play.
#
# as to cv2.VideoWriter, it doesn't seem to support H264 or any of the codecs eg TikTok requires,
# at least in the build you get with `pip install opencv-python-headless`; you also can't seem
# to have control eg over the pixel format (like yuv420p, see below)
#
# finally, Qt's media writer classes seem to rely on ffmpeg, same as imageio.
class MP4:
    def __init__(self, fname, width, height, fps):
        import av
        self.av = av
        self.output = av.open(fname, 'w', format='mp4')
        self.stream = self.output.add_stream('h264', str(fps))
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = 'yuv420p' # Windows Media Player eats this up unlike yuv444p
        self.stream.options = {'crf': '17'} # quite bad quality with smaller file sizes without this
    def write_frame(self, pixels):
        frame = self.av.VideoFrame.from_ndarray(pixels, format='rgb24')
        # without this reformat() call with both the format and the colorspace options, we get slightly
        # wrong colors (ffmpeg malfunctions similarly with the default settings and it's fixed by the -colorspace option,
        # though in that experiment I didn't try also specifying a yuv420p output; I don't know why we need to explicitly
        # convert the source to YUV in this code in addition to the converstion to the bt709 colorspace)
        frame = frame.reformat(format='yuv420p', dst_colorspace=self.av.video.reformatter.Colorspace.ITU709)
        packet = self.stream.encode(frame)
        self.output.mux(packet)
    def close(self):
        packet = self.stream.encode(None)
        self.output.mux(packet)
        self.output.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

interrupted = False
def interrupt_export():
    global interrupted
    interrupted = True

def check_if_interrupted():
    if interrupted:
        raise KeyboardInterrupt

def transpose_xy(image):
    return np.transpose(image, [1,0,2]) if len(image.shape)==3 else np.transpose(image, [1,0])

import cv2

def interruptible_export(movie):
    check_if_interrupted()

    def get_frame_images(i):
        transparent_frame = movie._blit_layers(movie.layers, i, transparent=True)
        frame = Surface((res.IWIDTH, res.IHEIGHT), color=BACKGROUND)
        frame.blit(transparent_frame, (0,0))
        check_if_interrupted()
        return frame, transparent_frame

    def write_transparent_png(fname, transparent_frame):
        transparent_pixels = transpose_xy(surf.pixels3d(transparent_frame))
        transparent_pixels = np.dstack([cv2.cvtColor(transparent_pixels, cv2.COLOR_RGB2BGR), transpose_xy(surf.pixels_alpha(transparent_frame))])

        cv2.imwrite(fname, transparent_pixels)
        check_if_interrupted()

    def write_opaque_png(fname, pixels):
        cv2.imwrite(fname, cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        check_if_interrupted()

    if len(movie.frames) == 1:
        # no animation - produce a single PNG image
        frame, transparent_frame = get_frame_images(0)
        write_transparent_png(movie.png_path(0), transparent_frame)
        pixels = transpose_xy(surf.pixels3d(frame))
        write_opaque_png(movie.still_png_path(), pixels)
        return

    assert FRAME_RATE==12
    opaque_ext = '.opaque.png'
    try:
        with MP4(movie.mp4_path(), res.IWIDTH, res.IHEIGHT, fps=24) as mp4_writer:
            for i in range(len(movie.frames)):
                frame, transparent_frame = get_frame_images(i)

                pixels = transpose_xy(surf.pixels3d(frame))
                check_if_interrupted()

                # append each frame twice at MP4 to get a standard 24 fps frame rate
                # (for GIFs there's less likelihood that something has a problem with
                # "non-standard 12 fps" (?))

                mp4_writer.write_frame(pixels)
                check_if_interrupted() 
                mp4_writer.write_frame(pixels)
                check_if_interrupted() 

                write_transparent_png(movie.png_path(i), transparent_frame)

                # non-transparent PNGs for the GIF generation, see also below
                write_opaque_png(movie.png_path(i)+opaque_ext, pixels)

        # we fill the background color rather than producing transparent GIFs for 2.5 reasons:
        # * GIF transparency is binary so you get ugly aliasing artifacts
        # * when you upload GIFs (eg to Twitter or WhatsApp), transparent pixels are filled with arbitrary background color (eg white or black) anyway
        # * gifski docs say that transparent GIFs are limited to 256 colors whereas non-transparent GIFs can actually have more; this is unlikely
        #   to be a big deal for us given our fairly restricted use of color but still.
        # so transparent GIFs are not as great as one might have hoped for. the sophisticated user knowing what they're doing gets transparent PNGs
        # that can be converted into any format, including transparent GIF/WebP/APNG. someone who just wants to get a GIF to upload is probably better
        # served by a WYSIWYG non-transparent GIF with the same background color they see when viewing the clip in Tinymation
 
        # FIXME proper path
        gifski = '..\\gifski\\gifski-win.exe' if on_windows else '../gifski/gifski-linux'
        os.system(f'{gifski} --width 1920 -r {FRAME_RATE} --quiet {movie.png_wildcard()+opaque_ext} --output {movie.gif_path()}')
 
    finally:
        # remove the non-transparent PNGs created for GIF generation
        for i in range(len(movie.frames)):
            try:
                os.unlink(movie.png_path(i)+opaque_ext)
            except:
                continue

    #print('done with',clipdir)

def export(movie):
    try:
        interruptible_export(movie)
    except KeyboardInterrupt:
        print('INTERRUPTED')
        pass
    except:
        import traceback
        traceback.print_exc()

def get_last_modified(filenames):
    f2mtime = {}
    for f in filenames:
        s = os.stat(f)
        f2mtime[f] = s.st_mtime
    return list(sorted(f2mtime.keys(), key=lambda f: f2mtime[f]))[-1]

def is_exported_png(f): return f.endswith('.png') and f != CURRENT_FRAME_FILE

class ExportProgressStatus:
    def __init__(self, clipdir, num_frames):
        self.clipdir = clipdir
        self.total = num_frames
    def update(self, live_clips):
        self.done = 0
        import re
        fmt = re.compile(r'frame([0-9]+)\.png')
        pngs = [f for f in os.listdir(self.clipdir) if is_exported_png(f)]
        if pngs:
            last = get_last_modified([os.path.join(self.clipdir, f) for f in pngs])
            m = fmt.match(os.path.basename(last))
            if m:
                 self.done = int(m.groups()[0]) + 1 # frame 3 being ready means 4 are done

_empty_frame = Frame('')

if on_windows:
    import winpath
    MY_DOCUMENTS = winpath.get_my_documents()
else:
    MY_DOCUMENTS = os.path.expanduser('~')

def set_wd(wd):
    global WD
    WD = wd
    if not os.path.exists(WD):
        os.makedirs(WD)
    
set_wd(os.path.join(MY_DOCUMENTS if MY_DOCUMENTS else '.', 'Tinymation'))


import datetime, time
def format_now(): return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

print('>>> STARTING',format_now())


import subprocess
import math
import io
import shutil

from PySide6.QtWidgets import QApplication, QWidget, QFileDialog, QLineEdit, QVBoxLayout, QPushButton, QHBoxLayout, QDialog, QMessageBox, QColorDialog
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QGuiApplication, QCursor, QPixmap
from PySide6.QtCore import Qt, QPoint, QEvent, QTimer, QCoreApplication, QEventLoop, QSize

app = QApplication(sys.argv)

import psutil

def is_already_running(lock_file_path):
    """Check if another instance of the program is running.
    This code is racy, strictly speaking, but you'd have to start the 2 processes
    so close to each other for the race to happen that it's not realistic for a human user.
    OTOH this avoids fiddling with real file locking or choosing a socket port or some such."""
    if os.path.exists(lock_file_path):
        try:
            # Read the PID from the lock file
            with open(lock_file_path, 'r') as f:
                pid = int(f.read().strip())

            # Check if the process with this PID is still running
            if psutil.pid_exists(pid):
                return True
            else:
                # PID exists but process is dead, remove stale lock file
                os.remove(lock_file_path)
                return False
        except (ValueError, OSError):
            # Invalid PID or file, assume stale lock file
            os.remove(lock_file_path)
            return False
    return False

def create_lock_file(lock_file_path):
    """Create a lock file with the current process ID."""

def show_already_running_dialog():
    """Display a modal dialog indicating another instance is running."""
    dialog = QMessageBox()
    dialog.setWindowTitle("Tinymation is already running!")
    dialog.setText("Tinymation is already running! Only one Tinymation window can run at a time.")
    dialog.setIcon(QMessageBox.Warning)
    dialog.setStandardButtons(QMessageBox.Ok)
    dialog.exec()

def create_lock_file():
    lockpath = os.path.join(WD, '.lock')

    if is_already_running(lockpath):
        show_already_running_dialog()
        sys.exit(1)

    with open(lockpath, 'w') as f:
        f.write(str(os.getpid()))

def delete_lock_file():
    try:
        os.unlink(os.path.join(WD, '.lock'))
    except:
        pass

create_lock_file()

#screen = pg.display.set_mode((800, 350*2), pg.RESIZABLE)
#screen = pg.display.set_mode((350, 800), pg.RESIZABLE)
#screen = pg.display.set_mode((1200, 350), pg.RESIZABLE)
#screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)
screen = Surface((1920, 1200), color=BACKGROUND)#pg.display.set_mode((0, 0), pg.FULLSCREEN)
#pg.display.flip()
#pg.display.set_caption("Tinymation")

#font = pg.font.Font(size=screen.get_height()//15)

FADING_RATE = 12
UNDRAWABLE = (220, 215, 190)
MARGIN = (220-80, 215-80, 190-80, 192)
MARGIN_BLENDED = [int((ch*MARGIN[-1] + ch2*(255-MARGIN[-1]))/255) for ch,ch2 in zip(MARGIN[:3], BACKGROUND)]
SELECTED = (220-80, 215-80, 190-80)
UNUSED = SELECTED
OUTLINE = (220-105, 215-105, 190-105)
PROGRESS = (192-45, 255-25, 192-45)
LAYERS_BELOW = (128,192,255)
LAYERS_ABOVE = (255,192,0)
WIDTH = 3 # the smallest width where you always have a pure pen color rendered along
# the line path, making our naive flood fill work well...
MEDIUM_ERASER_WIDTH = 5*WIDTH
BIG_ERASER_WIDTH = 20*WIDTH
PAINT_BUCKET_WIDTH = 3*WIDTH
CURSOR_SIZE = round(screen.get_width() * 0.055)
MAX_HISTORY_BYTE_SIZE = 1*1024**3
LIST_RECT_CORNER_RADIUS = round(screen.get_width() * 0.015)
SELECTION_CORNER_RADIUS = LIST_RECT_CORNER_RADIUS / 3

def list_rect(surface, rect, selected):
    inner_color, outer_color = OUTLINE, UNDRAWABLE
    radius = LIST_RECT_CORNER_RADIUS/3 if not selected else LIST_RECT_CORNER_RADIUS
    if selected:
        surf.rect(surface, outer_color, rect, 6, radius)
    surf.rect(surface, inner_color, rect, 4 if selected else 2, radius)
    surf.rect(surface, outer_color, rect, 1, radius)

print('clips read from, and saved to',WD)

# add tdiff() to printouts to see how many ms passed since the last call to tdiff()
prevts=time.time_ns()
def tdiff():
    global prevts
    now=time.time_ns()
    diff=(now-prevts)//10**6
    prevts = now
    return diff

from trace import trace

# interface with tinylib

def arr_base_ptr(arr): return arr.ctypes.data_as(ct.c_void_p)

def color_c_params(rgb):
    width, height, depth = rgb.shape
    assert depth == 3
    xstride, ystride, zstride = rgb.strides
    oft = 0
    assert xstride == 4 and zstride == 1, f'xstride={xstride}, ystride={ystride}, zstride={zstride}'
    ptr = ct.c_void_p(arr_base_ptr(rgb).value + oft)
    return ptr, ystride, width, height

def greyscale_c_params(grey, is_alpha=True, expected_xstride=1):
    width, height = grey.shape
    xstride, ystride = grey.strides
    assert (xstride == 4 and is_alpha) or (xstride == expected_xstride and not is_alpha), f'xstride={xstride} is_alpha={is_alpha}'
    ptr = arr_base_ptr(grey)
    return ptr, ystride, width, height

def make_color_int(rgba):
    r,g,b,a = rgba
    return r | (g<<8) | (b<<16) | (a<<24)

from tinylib_ctypes import tinylib, LayerParamsForMask, MaskAlphaParams, RangeFunc
import ctypes as ct

def rgba_array(surface):
    return surface._a

import cv2
def cv2_resize_surface(src, dst, inv_scale=None, best_quality=False):
    # if we pass the array as is, cv2.resize spends most of the time on converting
    # it to the layout it expects
    iattached = src.trans_unsafe()
    oattached = dst.trans_unsafe()

    iwidth, iheight = src.get_size()
    owidth, oheight = dst.get_size()

    if owidth < iwidth/2:
        method = cv2.INTER_AREA
        stat = surf.scale_inter_area_stat
    elif owidth > iwidth:
        method = cv2.INTER_CUBIC
        stat = surf.scale_inter_cubic_stat
    else:
        method = cv2.INTER_LINEAR if not best_quality else cv2.INTER_CUBIC
        stat = surf.scale_inter_linear_stat if not best_quality else surf.scale_inter_cubic_stat

    stat.start()
    if inv_scale is not None:
        cv2.resize(iattached, None, oattached, fx=inv_scale, fy=inv_scale, interpolation=method)
    else:
        cv2.resize(iattached, (owidth,oheight), oattached, interpolation=method)
    stat.stop(iwidth*iheight if iwidth>owidth else owidth*oheight)

def scaled_image_size(iwidth, iheight, width=None, height=None, inv_scale=None):
    assert width or height or inv_scale

    if inv_scale is not None:
        # from OpenCV (resize.cpp, cv::resize):
        #    dsize = Size(saturate_cast<int>(ssize.width*inv_scale_x),
        #                 saturate_cast<int>(ssize.height*inv_scale_y));
        # from fast_math.hpp:
        # template<> inline int saturate_cast<int>(double v)           { return cvRound(v); }
        width = round(iwidth * inv_scale)
        height = round(iheight * inv_scale)
        
    if not height:
        height = int(iheight * width / iwidth)
    if not width:
        width = int(iwidth * height / iheight)

    return width, height

def scale_image(surface, width=None, height=None, inv_scale=None, best_quality=False):
    width, height = scaled_image_size(surface.get_width(), surface.get_height(), width, height, inv_scale)

    if not best_quality and width < surface.get_width()//2 and height < surface.get_height()//2:
        return scale_image(scale_image(surface, surface.get_width()//2, surface.get_height()//2), width, height)

    ret = Surface((width, height), color=surf.COLOR_UNINIT)
    cv2_resize_surface(surface, ret, inv_scale, best_quality)
    ret.set_alpha(surface.get_alpha())
    #ret = surf.smoothscale(surface, (width, height))

    return ret

def minmax(v, minv, maxv):
    return min(maxv,max(minv,v))

def surf2cursor(surface, hotx, hoty):
  image = surface.qimage()
  pixmap = QPixmap.fromImage(image)
  return QCursor(pixmap, hotX=hotx, hotY=hoty)

def load_cursor(file, flip=False, size=CURSOR_SIZE, hot_spot=(0,1), min_alpha=192, edit=lambda x: (x, None), hot_spot_offset=(0,0)):
  surface = load_image(file)
  surface = scale_image(surface, size, size*surface.get_height()/surface.get_width(), best_quality=True)#surf.scale(surface, (CURSOR_SIZE, CURSOR_SIZE))
  if flip:
      surface = surf.flip(surface, True, True)
  non_transparent_surface = surface.copy()
  alpha = surf.pixels_alpha(surface)
  alpha[:] = np.minimum(alpha, min_alpha)
  del alpha
  surface, hot = edit(surface)
  if hot is None:
      hotx = minmax(int(hot_spot[0] * surface.get_width()) + hot_spot_offset[0], 0, surface.get_width()-1)
      hoty = minmax(int(hot_spot[1] * surface.get_height()) + hot_spot_offset[1], 0, surface.get_height()-1)
  else:
      hotx, hoty = hot
  #return pg.cursors.Cursor((hotx, hoty), surface), non_transparent_surface
  return surf2cursor(surface, hotx, hoty), non_transparent_surface

def add_circle(image, radius, offset=(0,1), color=(255,0,0,128), outline_color=(0,0,0,128)):
    new_width = max(image.get_width(), radius + round(image.get_width()*(1-offset[0])))
    new_height = max(image.get_height(), radius + round(image.get_height()*offset[1]))
    result = Surface((new_width, new_height))
    xoffset = round(offset[0]*image.get_width())
    yoffset = round(offset[1]*image.get_height())
    radius -= .5
    surf.filled_circle(result, radius, yoffset, radius, outline_color)
    surf.filled_circle(result, radius, yoffset, radius-WIDTH+1, color)
    result.blit(image, (radius-xoffset, 0))
    return result, (radius, yoffset)

pen_cursor = load_cursor('pen.png', size=int(CURSOR_SIZE*1.3), hot_spot=(0.02,0.97))
pen_cursor = (pen_cursor[0], load_image('pen-tool.png'))
pencil_cursor = load_cursor('pencil.png', size=int(CURSOR_SIZE*1.3), hot_spot=(0.02,0.97))
pencil_cursor = (pencil_cursor[0], load_image('pencil-tool.png'))
tweezers_cursor = load_cursor('tweezers.png', size=int(CURSOR_SIZE*1.5), hot_spot=(0.03,0.97))
tweezers_cursor = (tweezers_cursor[0], load_image('tweezers-tool.png'))
eraser_cursor = load_cursor('eraser.png', size=int(CURSOR_SIZE*0.7), hot_spot=(0.02,0.9))
eraser_cursor = (eraser_cursor[0], load_image('eraser-tool.png'))
eraser_medium_cursor = load_cursor('eraser.png', size=int(CURSOR_SIZE), edit=lambda s: add_circle(s, MEDIUM_ERASER_WIDTH//2, offset=(0.02,0.9)), hot_spot_offset=(MEDIUM_ERASER_WIDTH//2,-MEDIUM_ERASER_WIDTH//2))
eraser_medium_cursor = (eraser_medium_cursor[0], eraser_cursor[1])
eraser_big_cursor = load_cursor('eraser.png', size=int(CURSOR_SIZE*1.5), edit=lambda s: add_circle(s, BIG_ERASER_WIDTH//2, offset=(0.02,0.9)), hot_spot_offset=(BIG_ERASER_WIDTH//2,-BIG_ERASER_WIDTH//2))
eraser_big_cursor = (eraser_big_cursor[0], eraser_cursor[1])
needle_cursor = load_cursor('needle.png', size=int(CURSOR_SIZE), hot_spot=(0.02,0.97))
flashlight_cursor = needle_cursor
flashlight_cursor = (flashlight_cursor[0], load_image('needle-tool.png')) 
paint_bucket_cursor = (load_cursor('paint_bucket.png', size=int(CURSOR_SIZE*1.2))[1], load_image('splash-0.png')) #FIXME
blank_page_cursor = load_cursor('sheets.png', hot_spot=(0.5, 0.5))
garbage_bin_cursor = load_cursor('garbage.png', hot_spot=(0.5, 0.5))
zoom_cursor = (load_cursor('zoom.png', hot_spot=(0.75, 0.5), size=int(CURSOR_SIZE*1.5))[0], load_image('zoom-tool.png'))
finger_cursor = load_cursor('finger.png', hot_spot=(0.85, 0.17))

# for locked screen
empty_cursor = surf2cursor(Surface((10,10)), 0, 0)

# set_cursor can fail on some machines so we don't count on it to work.
# we set it early on to "give a sign of life" while the window is black;
# we reset it again before entering the event loop.
# if the cursors cannot be set the selected tool can still be inferred by
# the darker background of the tool selection button.
prev_cursor = None
curr_cursor = None
def try_set_cursor(c):
    try:
        global curr_cursor
        global prev_cursor
        widget.setCursor(c)
        prev_cursor = curr_cursor
        curr_cursor = c
    except Exception as e:
        print('Failed to set cursor',e)
        pass

def restore_cursor():
    try_set_cursor(prev_cursor)

def bounding_rectangle_of_a_boolean_mask(mask):
    # Sum along the vertical and horizontal axes
    vertical_sum = np.sum(mask, axis=1)
    if not np.any(vertical_sum):
        return None
    horizontal_sum = np.sum(mask, axis=0)

    minx, maxx = np.where(vertical_sum)[0][[0, -1]]
    miny, maxy = np.where(horizontal_sum)[0][[0, -1]]

    return minx, maxx, miny, maxy

class EditablePenLine:
    def __init__(self, points, start_time=None, closed=False):
        self.points = points
        self.closed = closed
        self.frame_without_line = None
        self.start_time = start_time

class HistoryItemBase:
    def __init__(self, restore_pos_before_undo=True):
        self.restore_pos_before_undo = restore_pos_before_undo
        self.pos_before_undo = movie.pos
        self.layer_pos_before_undo = movie.layer_pos
        self.editable_pen_line = None
    def is_drawing_change(self): return False
    def from_curr_pos(self): return self.pos_before_undo == movie.pos and self.layer_pos_before_undo == movie.layer_pos
    def byte_size(history_item): return 128
    def nop(history_item): return False
    def bounding_rect(self): return None
    def make_undone_changes_visible(self):
        needed_change = False
        if not self.restore_pos_before_undo:
            return needed_change

        if movie.pos != self.pos_before_undo or movie.layer_pos != self.layer_pos_before_undo:
            movie.seek_frame_and_layer(self.pos_before_undo, self.layer_pos_before_undo)
            needed_change = True

        rect = self.bounding_rect()
        if rect is not None:
            da = layout.drawing_area()
            l, b, w, h = da.rois(just_the_misaligned_frame_roi=True)
            r, t = l+w, b+h
            l1, b1, r1, t1 = rect
            il, ib, ir, it = max(l,l1), max(b,b1), min(r,r1), min(t,t1)
            if ir-il <= 0 or it-ib <= 0 or (ir-il)*(it-ib) < min(0.15*(r1-l1)*(t1-b1), (WIDTH*5)**2): # undone change nearly invisible in the drawing area
                da.reset_zoom_pan_params()
                needed_change = True

        return needed_change

class HistoryItem(HistoryItemBase):
    def __init__(self, surface_id, bbox=None):
        HistoryItemBase.__init__(self)
        self.surface_id = surface_id
        if not bbox:
            # perhaps allocating from a pool and copying via rgba_array would be faster but not sure it would matter
            self.saved_surface = self.curr_surface().copy()
            self.minx = 10**9
            self.miny = 10**9
            self.maxx = -10**9
            self.maxy = -10**9
            self.optimized = False
        else:
            self.minx, self.miny, self.maxx, self.maxy = bbox
            self.saved_surface = self.curr_surface().subsurface(self._subsurface_bbox()).copy()
            self.optimized = True

    def _subsurface_bbox(self): return (self.minx, self.miny, self.maxx-self.minx+1, self.maxy-self.miny+1)
    def bounding_rect(self):
        if self.optimized:
            return self.minx, self.miny, self.maxx+1, self.maxy+1
        return 0, 0, res.IWIDTH, res.IHEIGHT
    def is_drawing_change(self): return True
    def curr_surface(self):
        return movie.edit_curr_frame().surf_by_id(self.surface_id)
    def nop(self):
        return self.saved_surface is None
    def undo(self):
        if self.nop():
            return

        if self.pos_before_undo != movie.pos or self.layer_pos_before_undo != movie.layer_pos:
            print(f'WARNING: HistoryItem at the wrong position! should be {self.pos_before_undo} [layer {self.layer_pos_before_undo}], but is {movie.pos} [layer {movie.layer_pos}]')
        movie.seek_frame_and_layer(self.pos_before_undo, self.layer_pos_before_undo) # we should already be here, but just in case (undoing in the wrong frame is a very unfortunate bug...)

        redo = HistoryItem(self.surface_id, (self.minx, self.miny, self.maxx, self.maxy) if self.optimized else None)
        redo.editable_pen_line = self.editable_pen_line

        frame = self.curr_surface()
        self.copy_saved_subsurface_into(frame)
        return redo

    def copy_saved_subsurface_into(self, frame):
        if self.optimized:
            frame = frame.subsurface(self._subsurface_bbox())
        
        rgba_array(frame)[:] = rgba_array(self.saved_surface)

    def optimize(self, bbox=None):
        if self.optimized:
            return

        if bbox:
            self.minx, self.miny, self.maxx, self.maxy = bbox
        else:
            mask = np.any(rgba_array(self.saved_surface) != rgba_array(self.curr_surface()), axis=2)
            brect = bounding_rectangle_of_a_boolean_mask(mask)

            if brect is None: # this can happen eg when drawing lines on an already-filled-with-lines area
                self.saved_surface = None
                return

            self.minx, self.maxx, self.miny, self.maxy = [int(c) for c in brect]

        # TODO: test that this actually reduces memory consumption
        self.saved_surface = self.saved_surface.subsurface(self._subsurface_bbox()).copy()
        self.optimized = True

    def __str__(self):
        return f'HistoryItem(pos={self.pos}, rect=({self.minx}, {self.miny}, {self.maxx}, {self.maxy}))'

    def byte_size(self):
        return self.saved_surface.get_width() * self.saved_surface.get_height() * 4 if not self.nop() else 0

class HistoryItemSet(HistoryItemBase):
    def __init__(self, items):
        HistoryItemBase.__init__(self)
        self.items = [item for item in items if item is not None]
    def is_drawing_change(self):
        for item in self.items:
            if not item.is_drawing_change():
                return False
        return True
    def bounding_rect(self):
        rect = None
        for item in self.items:
            irect = item.bounding_rect()
            if irect is not None:
                if rect is None:
                    rect = irect
                else:
                    l1, b1, r1, t1 = rect
                    l2, b2, r2, t2 = irect
                    rect = min(l1,l2), min(b1,b2), max(r1,r2), max(t1,t2)
        return rect
    def nop(self):
        for item in self.items:
            if not item.nop():
                return False
        return True
    def undo(self):
        return HistoryItemSet(list(reversed([item.undo() for item in self.items])))
    def optimize(self, bbox=None):
        for item in self.items:
            item.optimize(bbox)
        self.items = [item for item in self.items if not item.nop()]
    def byte_size(self):
        return sum([item.byte_size() for item in self.items])
    def make_undone_changes_visible(self):
        for item in self.items:
            if item.make_undone_changes_visible():
                return True

def scale_and_preserve_aspect_ratio(w, h, width, height):
    if width/height > w/h:
        scaled_width = w*height/h
        scaled_height = h*scaled_width/w
    else:
        scaled_height = h*width/w
        scaled_width = w*scaled_height/h
    return round(scaled_width), round(scaled_height)

class LayoutElemBase:
    def __init__(self): self.redraw = True
    def init(self): pass
    def hit(self, x, y): return True
    def modify(self): pass
    def draw(self): pass
    def highlight_selection(self): pass
    def on_mouse_down(self, x, y): pass
    def on_mouse_move(self, x, y): pass
    def on_mouse_up(self, x, y): pass
    def on_history_timer(self): pass

class Button(LayoutElemBase):
    def __init__(self):
        LayoutElemBase.__init__(self)
        self.button_surface = None
        self.only_hit_non_transparent = False
    def draw(self, rect, cursor_surface):
        left, bottom, width, height = rect
        _, _, w, h = cursor_surface.get_rect()
        scaled_width, scaled_height = scale_and_preserve_aspect_ratio(w, h, width, height)
        if not self.button_surface:
            surface = scale_image(cursor_surface, scaled_width, scaled_height, best_quality=True)
            self.button_surface = surface
        self.screen_left = int(left+(width-scaled_width)/2)
        self.screen_bottom = int(bottom+height-scaled_height)
        screen.blit(self.button_surface, (self.screen_left, self.screen_bottom))
    def hit(self, x, y, rect=None):
        if not self.only_hit_non_transparent:
            return True
        if rect is None:
            rect = self.rect
        if not self.button_surface:
            return False
        left, bottom, width, height = rect
        try:
            alpha = self.button_surface.get_at((x-self.screen_left, y-self.screen_bottom))[3]
        except:
            return False
        return alpha > 0

locked_image = load_image('lock-mask.png')
invisible_image = load_image('eye_shut.png')
def curr_layer_locked():
    effectively_locked = movie.curr_layer().locked or not movie.curr_layer().visible
    if effectively_locked: # invisible layers are effectively locked but we show it differently
        da = layout.drawing_area()
        reason_image = locked_image if movie.curr_layer().locked else invisible_image
        tw, th = scale_and_preserve_aspect_ratio(reason_image.get_width(), reason_image.get_height(), 3*da.iwidth/4, 3*da.iheight/4)
        reason_image = scale_image(reason_image, tw, th, best_quality=True)
        # make the surface big enough to have pixels for subsurface() whatever starting_point is in DrawingArea.draw
        full_fading_mask = Surface((da.subsurface.get_width() + da.lmargin + da.rmargin, da.subsurface.get_height() + da.ymargin*2)) #new_frame()
        fading_mask = full_fading_mask.subsurface((0, 0, da.iwidth+da.lmargin*2, da.iheight+da.ymargin*2)) # this is lmargin*2 instead of lmargin+rmargin deliberately
        # - for vertical layouts we draw within the left side of the drawing area which is not partially covered with timeline and layers area;
        # blits() wouldn't work for us to draw the entire surface if we veered into the right margin area
        fading_mask.blit(reason_image, ((fading_mask.get_width()-reason_image.get_width())//2, (fading_mask.get_height()-reason_image.get_height())//2))
        full_fading_mask.set_alpha(192)
        da.set_fading_mask(full_fading_mask, prescaled_fading_mask=True)
        da.fade_per_frame = 192/(FADING_RATE*7)
    return effectively_locked

def find_nearest(array, center_x, center_y, value):
    '''Find the coordinates of the element having the given value that is closest to the given center point.'''
    ones_coords = np.where(array == value)
    if len(ones_coords[0]) == 0:
        return None
    points = np.column_stack(ones_coords)
    distances = np.sqrt((points[:, 0] - center_x)**2 + 
                       (points[:, 1] - center_y)**2)
    nearest_idx = np.argmin(distances)
    return tuple(points[nearest_idx])

class PenTool(Button):
    def __init__(self, eraser=False, soft=False, width=WIDTH, zoom_changes_pixel_width=True, rgb=None):
        Button.__init__(self)
        self.prev_drawn = None
        self.color = BACKGROUND if eraser else PEN
        self.eraser = eraser
        self.soft = soft
        self.width = width
        self.zoom_changes_pixel_width = zoom_changes_pixel_width
        self.circle_width = (width//2)*2
        self.smooth_dist = 40
        self.points = []
        self.polyline = []
        self.lines_array = None
        self.rect = np.zeros(4, dtype=np.int32)
        self.region = arr_base_ptr(self.rect)
        self.bbox = None
        self.history_time_period = 1000
        self.timer = None
        self.patching = False
        self.rgb = rgb

    def brush_flood_fill_color_based_on_mask(self):
        mask_ptr, mask_stride, width, height = greyscale_c_params(self.pen_mask, is_alpha=False)
        flood_code = 2

        color = surf.pixels3d(movie.edit_curr_frame().surf_by_id('color'))
        color_ptr, color_stride, color_width, color_height = color_c_params(color)
        assert color_width == width and color_height == height
        # the RGB values of transparent colors can actually matter in some (stupid) contexts - pasting into
        # some apps with no transparency support exposes these RGB values...
        # TODO: make sure we fill with BACKGROUND in all flows
        new_color_value = make_color_int(self.bucket_color if self.bucket_color else BACKGROUND+(0,))

        tinylib.brush_flood_fill_color_based_on_mask(self.brush, color_ptr, mask_ptr, color_stride, mask_stride, 0, flood_code, new_color_value)

    def find_bucket_color(self, x, y):
        w = 10
        h = 10
        while True:
            area = max(0, round(x-w/2)), max(0, round(y-h/2)), min(res.IWIDTH, round(x+w/2)), min(res.IHEIGHT, round(y+h/2))
            if area == (0, 0, res.IWIDTH, res.IHEIGHT):
                break
            l,b,r,t = area
            tdiff()
            nearest_fully_transparent = find_nearest(self.lines_array[l:r,b:t], x, y, 0)
            if nearest_fully_transparent is None:
                # no transparent pixels found - search in a larger area (we could just search the entire
                # image but it would be slower)
                w *= 2
                h *= 2
                continue
            nx, ny = nearest_fully_transparent
            self.bucket_color = movie.curr_frame().surf_by_id('color').get_at((nx+l,ny+b))
            break

    def init_brush(self, x, y, smoothDist=None, dry=False, paintWithin=None):
        if smoothDist is None:
            smoothDist = self.smooth_dist
        ptr, ystride, width, height = greyscale_c_params(self.lines_array)
        if self.rgb:
            color = surf.pixels3d(movie.edit_curr_frame().surf_by_id('lines'))
            color_ptr, color_stride, color_width, color_height = color_c_params(color)
            ptr = color_ptr
        lineWidth = self.width if not self.zoom_changes_pixel_width else self.width*layout.drawing_area().xscale
        self.brush = tinylib.brush_init_paint(x, y, layout.event_time, layout.pressure, lineWidth, smoothDist, dry,
                1 if self.eraser else 0, 1 if self.soft else 0, ptr, width, height, 4, ystride, arr_base_ptr(paintWithin) if paintWithin is not None else 0)
        if self.rgb:
            tinylib.brush_set_rgb(self.brush, (ct.c_uint8*3)(*self.rgb))

    def on_mouse_down(self, x, y):
        if curr_layer_locked():
            return
        self.patching = ctrl_is_pressed()
        if self.patching:
            NeedleTool().on_mouse_down(x,y)
            return
        self.points = []
        self.polyline = []
        self.bucket_color = None
        self.lines_array = surf.pixels_alpha(movie.edit_curr_frame().surf_by_id('lines'))

        cx, cy = layout.drawing_area().xy2frame(x, y)
        self.init_brush(cx, cy)
        if self.eraser:
            self.pen_mask = self.lines_array == 255
            self.find_bucket_color(cx, cy)
            self.brush_flood_fill_color_based_on_mask()

        self.new_history_item()

        self.prev_drawn = (x,y) # Krita feeds the first x,y twice - in init-paint and in paint, here we do, too
        self.on_mouse_move(x,y)
        if self.eraser: # we split eraser gestures into 1-second parts since sometimes you erase for a lot of time
            # without ever putting down the eraser and at some point erase too much and you don't want to undo all
            # that time spend erasing. with drawing it's less like it (undoing a part of the line seems less likely
            # to be what you want and you naturally break drawing into separate "gestures" making sense "as a whole";
            # if we do decide to split lines into parts for undo purposes, we should not doing when not layout.subpixel
            # since this doesn't work with smoothing)
            self.set_history_timer()

    def set_history_timer(self):
        if self.timer is None:
            self.timer = QTimer(widget)
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(self.on_history_timer)
        self.timer.start(self.history_time_period)

    def new_history_item(self):
        self.bbox = (1000000, 1000000, -1, -1)
        self.lines_history_item = HistoryItem('lines')
        if self.eraser:
            self.color_history_item = HistoryItem('color')

    def update_bbox(self):
        xmin, ymin, xmax, ymax = self.bbox
        rxmin, rymin, rxmax, rymax = self.rect
        self.bbox = (min(xmin, rxmin), min(ymin, rymin), max(xmax, rxmax), max(ymax, rymax))

        xmin, ymin, xmax, ymax = self.short_term_bbox
        self.short_term_bbox = (min(xmin, rxmin), min(ymin, rymin), max(xmax, rxmax), max(ymax, rymax))

    def smooth_line(self):
        assert not self.eraser and not self.soft
        try:
            px, py = bspline_interp(self.points, smoothing=len(self.points)/(layout.drawing_area().zoom*2))
        except:
            return # if we can't smooth the line (eg not enough points), NP, we'll just keep the raw input

        self.lines_history_item.undo()
        self.new_history_item()

        self.draw_line(list(zip(px,py)), smoothDist=0) # don't smooth, bspline_interp already did

    def end_paint(self, nsamples=0, get_sample2polyline=False):
        tinylib.brush_end_paint(self.brush, self.region)
        self.update_bbox()

        polyline_length = tinylib.brush_get_polyline_length(self.brush)
        self.polyline = np.zeros((polyline_length, 2), dtype=float, order='F')
        polyline_x = self.polyline[:, 0]
        polyline_y = self.polyline[:, 1]
        tinylib.brush_get_polyline(self.brush, polyline_length, arr_base_ptr(polyline_x), arr_base_ptr(polyline_y), 0, 0)

        if get_sample2polyline:
            self.sample2polyline = np.empty(nsamples, dtype=np.int32)
            tinylib.brush_get_sample2polyline(self.brush, nsamples, arr_base_ptr(self.sample2polyline))

        tinylib.brush_free(self.brush)
        self.brush = 0

    def draw_line(self, xys, zoom=1, smoothDist=None, dry=False, paintWithin=None, get_sample2polyline=False, closed=False):
        assert not self.eraser
        if len(xys) < 2:
            return

        # Convert xys to numpy array if it's a list of tuples
        if isinstance(xys, list):
            arr = np.array(xys, order='F', dtype=float)
        else:
            arr = xys

        x0, y0 = arr[0]
        self.init_brush(x0, y0, smoothDist=smoothDist, dry=dry, paintWithin=paintWithin)
        if closed:
            tinylib.brush_set_closed(self.brush)

        xarr = arr[1:, 0]
        yarr = arr[1:, 1]

        tinylib.brush_paint(self.brush, len(xys)-1, arr_base_ptr(xarr), arr_base_ptr(yarr), 0, 0, zoom, self.region)
        self.update_bbox()

        self.end_paint(nsamples=len(xys)+int(closed), get_sample2polyline=get_sample2polyline)

        self.points = xys

    def on_mouse_up(self, x, y):
        if self.patching or curr_layer_locked():
            return

        if self.timer is not None and self.timer.isActive():
            self.timer.stop()

        movie.edit_curr_frame()
    
        self.end_paint()

        # it doesn't sound good to smooth "mice erasers"
        # because while nominally pens and erasers are basically the same, you draw with a pen to get nice lines,
        # so you prefer them smoothed rather than getting the ugly mouse artefacts, but you use erasers to get
        # rid of what you're erasing, so you don't want to aim the eraser paintakingly at something and then
        # suddenly have slightly different things erased because of smoothing when you lift the pen (not to mention
        # the "ripple effect" on color the way our erasers work)
        if not layout.subpixel and not self.eraser and not self.soft:
            self.smooth_line()

        self.prev_drawn = None

        self.save_history_item()

        self.lines_array = None

    def save_history_item(self):
        if self.bbox[-1] >= 0:
            history_item = HistoryItemSet([self.lines_history_item, self.color_history_item]) if self.eraser else self.lines_history_item
            history_item.optimize(self.bbox)

            if not self.soft and not self.eraser: # pen rather than pencil or eraser
                history_item.editable_pen_line = EditablePenLine(self.polyline) # can be edited with TweezersTool

            history.append_item(history_item)

    def on_history_timer(self):
        self.save_history_item()
        self.new_history_item()
        self.set_history_timer()

    def on_mouse_move(self, x, y):
        if self.patching or curr_layer_locked():
            return

        drawing_area = layout.drawing_area()
        cx, cy = drawing_area.xy2frame(x, y)

        # no idea why this happens for fast pen motions, but it's been known to happen - we see the first coordinate repeated for some reason
        # note that sometimes you get something close but not quite equal to the first coordinate and it's clearly wrong because it's an outlier
        # relatively to the rest of the points; not sure if we should try to second-guess the input device enough to handle it...
        if len(self.points) < 6 and (cx, cy) in self.points:
            return

        self.points.append((cx,cy))

        if self.prev_drawn:
            self.short_term_bbox = (1000000, 1000000, -1, -1)
            xarr = np.array([cx], dtype=float)
            yarr = np.array([cy], dtype=float)
            tarr = np.array([layout.event_time], dtype=float)
            parr = np.array([min(layout.pressure,0.7) if layout.subpixel else 0.35], dtype=float)
            tinylib.brush_paint(self.brush, 1, *[arr_base_ptr(arr) for arr in [xarr, yarr, tarr, parr]], drawing_area.xscale, self.region)
            self.update_bbox()

            if self.short_term_bbox[-1] >= 0:
                layout.drawing_area().draw_region(self.short_term_bbox)
            
        self.prev_drawn = (x,y) 

def polyline_corners(points, curvature_threshold=1, peak_distance=15):
    # Fit a B-spline to the polyline
    points = np.column_stack([points[:,0],points[:,1]]) # array layout correction
    tck, u = splprep(points, smoothing=0)

    # Evaluate the spline at the u values returned by splprep
    dx, dy = splev(u, tck, der=1)  # First derivatives at u
    ddx, ddy = splev(u, tck, der=2)  # Second derivatives at u

    # Compute curvature at u values
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**1.5
    curvature = numerator / denominator 

    peaks = np.empty(len(curvature), dtype=np.uint8)
    tinylib.find_peaks(arr_base_ptr(peaks), arr_base_ptr(curvature), len(curvature), curvature_threshold, peak_distance)
    return peaks

def smooth_polyline(closed, points, focus, prev_closest_to_focus_idx=-1, threshold=30, smoothness=0.6, pull_strength=0.5, num_neighbors=1, max_endpoint_dist=30, zero_endpoint_dist_start=5, corner_stiffness=1, corner_vec=None):
    xarr = points[:, 0]
    yarr = points[:, 1]
    
    new_arr = np.zeros(points.shape, order='F', dtype=float)
    newx = new_arr[:, 0]
    newy = new_arr[:, 1]

    first_diff = np.zeros(1, dtype=np.int32)
    last_diff = np.zeros(1, dtype=np.int32)

    if corner_vec is None:
        corner_vec = polyline_corners(points)

    closest_idx = tinylib.smooth_polyline(closed, len(xarr), *[arr_base_ptr(a) for a in [newx,newy,xarr,yarr]], focus[0], focus[1],
            arr_base_ptr(first_diff), arr_base_ptr(last_diff), prev_closest_to_focus_idx,
            arr_base_ptr(corner_vec), corner_stiffness,
            threshold, smoothness, pull_strength, num_neighbors, max_endpoint_dist, zero_endpoint_dist_start)

    return new_arr, first_diff[0], last_diff[0]+1, closest_idx

def points_bbox(xys, margin):
    xs = [xy[0] for xy in xys]
    ys = [xy[1] for xy in xys]
    return min(xs)-margin, min(ys)-margin, max(xs)+margin, max(ys)+margin

def simplify_polyline(points, threshold, remap_idx=None):
    if len(points) < 100:
        return points, remap_idx if (remap_idx is not None and remap_idx < len(points) and remap_idx >= 0) else None
        
    remapped_idx = None
    result = [points[0]]
    if remap_idx == 0:
        remapped_idx = 0

    for i in range(1, len(points)-1):
        if remap_idx == i:
            remapped_idx = len(result)

        prev = points[i-1]
        curr = points[i]
        next = points[i+1]
        
        dist_prev = ((curr[0] - result[-1][0])**2 + (curr[1] - result[-1][1])**2)**0.5
        dist_next = ((curr[0] - next[0])**2 + (curr[1] - next[1])**2)**0.5
        
        # throw a point away if it's close enough to both neighbors, or if it's identical to at least one of them
        if (dist_prev >= threshold or dist_next >= threshold) and min(dist_prev, dist_next)>0:
            result.append(curr)
            
    result.append(points[-1])
    return result, remapped_idx

class TweezersTool(Button):
    def __init__(self):
        Button.__init__(self)
        self.editable_pen_line = None

    def on_mouse_down(self, x, y):
        if ctrl_is_pressed():
            try_to_close_the_last_editable_line(*layout.drawing_area().xy2frame(x,y))
            return

        last_item = history.last_item()
        if last_item:
            self.editable_pen_line = last_item.editable_pen_line
        if self.editable_pen_line is None:
            return

        self.lines = movie.edit_curr_frame().surf_by_id('lines')
        self.frame_without_line = self.editable_pen_line.frame_without_line
        if self.frame_without_line is None: # the first time we edit a line, we create
            # a surface without the line. we do it here rather than at PenTool.on_mouse_up
            # to avoid slowing down repeated pen use without editing
            self.frame_without_line = self.lines.copy()
            last_item.copy_saved_subsurface_into(self.frame_without_line)

        pen = TOOLS['pen'].tool
        pen.lines_array = surf.pixels_alpha(self.lines)

        self.history_item = HistoryItem('lines')

        self.rgba_lines = rgba_array(self.lines)
        self.rgba_frame_without_line = rgba_array(self.frame_without_line)
    
        self.corners = polyline_corners(self.editable_pen_line.points)

        self.prev_closest_to_focus_idx = -1

    def on_mouse_up(self, x, y):
        if self.editable_pen_line is None:
            return
        
        movie.edit_curr_frame()
        self.history_item.optimize()
        self.history_item.editable_pen_line = self.editable_pen_line
        history.append_item(self.history_item)

        pen = TOOLS['pen'].tool
        self.lines = None
        pen.lines_array = None
        self.editable_pen_line = None
        self.history_item = None

        self.rgba_lines = None
        self.rgba_frame_without_line = None

    def on_mouse_move(self, x, y):
        if self.editable_pen_line is None:
            return

        start_time = time.time_ns() / 1000000
        if self.editable_pen_line.start_time is not None:
            # we're measuring age using the timestamps when
            age = start_time - self.editable_pen_line.start_time
            if age > 10:
                self.editable_pen_line.start_time = start_time
                return # for very long lines we can't avoid a slowdown,
            # and it's not trivial to know when (it's not just a question of line length
            # but of how much the line intersects the region where a change happened and so how
            # much repainting it takes) - so we ignore old events

        #self.draw_corners(red=0)

        drawing_area = layout.drawing_area()
        cx, cy = drawing_area.xy2frame(x, y)
        pen = TOOLS['pen'].tool

        old_points = self.editable_pen_line.points
        closed = self.editable_pen_line.closed

        p = layout.pressure
        sq = (1+p)**5

        dist_thresh=(15*sq)/drawing_area.zoom
        neighbors=1 + math.floor(p*10)#(1-(1-p)**2)*10)

        endpoint_dist = 15

        new_points, first_diff, last_diff, closest_idx = smooth_polyline(closed, old_points, (cx,cy), self.prev_closest_to_focus_idx,
                                                                         threshold=dist_thresh, pull_strength=p, num_neighbors=neighbors, max_endpoint_dist=endpoint_dist,
                                                                         corner_stiffness=min(1,1.7-p*2), corner_vec=self.corners)

        if first_diff < 0:
            assert last_diff == len(old_points)+1
            return # no changes - nothing to do

        if last_diff <= first_diff: # wraparound
            assert np.all(old_points[last_diff:first_diff] == new_points[last_diff:first_diff])
            assert closed, f'{first_diff=} {last_diff=}'
            shift = len(old_points) - first_diff
            new_points = np.roll(new_points, shift, axis=0)
            old_points = np.roll(old_points, shift, axis=0)
            self.corners = np.roll(self.corners, shift)
            last_diff += shift
            first_diff = 0
            closest_idx = (closest_idx + shift) % len(old_points)

        # we allow ourselves the use of list (and Python code not calling into C) for the modified points, of which there are few;
        # the bulk of the points, of which there can be many, we manage as numpy arrays and process in C
        simplified_new_points, simplified_closest_idx = simplify_polyline(list(new_points[first_diff:last_diff]),1,closest_idx-first_diff)
        changed_old_points = list(old_points[first_diff:last_diff])

        if simplified_closest_idx is not None: # pretty sure that this "if" should ~always be true but we have code for when it isn't, in terms of index remapping it sounds correct
            simplified_closest_idx += first_diff
            self.prev_closest_to_focus_idx = simplified_closest_idx
        elif closest_idx >= last_diff: # closest point is after last diff 
            self.prev_closest_to_focus_idx = closest_idx - (len(changed_old_points)-len(simplified_new_points))
        else: # before first_diff 
            self.prev_closest_to_focus_idx = closest_idx

        simplified_new_points_array = np.array(simplified_new_points, dtype=float)
        assert len(self.corners) == len(old_points), f'{len(self.corners)=} {len(old_points)=} {first_diff=} {last_diff=} {len(simplified_new_points)=}'
        self.update_corners(old_points, simplified_new_points_array, first_diff, last_diff)

        new_points = np.asfortranarray(np.concatenate((old_points[:first_diff], simplified_new_points_array, old_points[last_diff:])))
        assert len(self.corners) == len(new_points), f'{len(self.corners)=} {len(old_points)=} {first_diff=} {last_diff=} {len(simplified_new_points)=}'

        affected_bbox = points_bbox(simplified_new_points + changed_old_points + list(old_points[first_diff-1:first_diff]) + list(old_points[last_diff:last_diff+1]), WIDTH*4)

        minx, miny, maxx, maxy = [round(c) for c in affected_bbox]
        minx, miny = res.clip(minx, miny)
        maxx, maxy = res.clip(maxx, maxy)
        self.rgba_lines[minx:maxx,miny:maxy] = self.rgba_frame_without_line[minx:maxx,miny:maxy]

        paintWithin = np.array([minx, miny, maxx, maxy],dtype=np.int32)

        pen.draw_line(new_points, smoothDist=0, paintWithin=paintWithin, get_sample2polyline=True, closed=closed)

        # the corners in pen.polyline are roughly where they were in new_points, shifted by sample2polyline
        # (this is inaccurate both because sample2polyline isn't currently very accurate and because it assumes
        # thigns about the curvature of the smoothed pen.polyline which might not be true, but it seems to work well
        # enough in practice. another approach would be to look for new corners (what update_corners currently does)
        # in pen.polyline rather than in simplified_new_points; this would probably be slightly more accurate since
        # currently we're assuming that fitpack smoothing works a lot like Krita-like brush smoothing and maybe
        # sometimes it doesn't.) of course even more sensible might have been to compute curvature in the brush
        # code directly without relying on fitpack
        polylen = len(pen.polyline)-int(closed)
        self.update_indexes(pen.sample2polyline, polylen)

        self.editable_pen_line = EditablePenLine(pen.polyline[:polylen], start_time, closed=closed)
        self.editable_pen_line.frame_without_line = self.frame_without_line

        #self.draw_corners(red=255)

        layout.drawing_area().draw_region(affected_bbox)
        # if you call draw_corners you might want to call this instead of the draw_region above for debugging:
        #layout.drawing_area().draw_region(points_bbox(list(old_points)+list(pen.polyline), WIDTH*4))

    def draw_corners(self,red):
        def draw_point(i,ch,val):
            x,y = self.editable_pen_line.points[i]
            self.rgba_lines[int(x),int(y),ch] = val
        for i,isc in enumerate(self.corners):
            if isc:
                draw_point(i,0,red)
        draw_point(self.prev_closest_to_focus_idx, 1, red)

    def update_corners(self, old_points, simplified_new_points, first_diff, last_diff):
        '''we're making an effort to only recompute corners at the changed part of the polyline rather than refit a bspline to the whole thing at every step'''
        distance = 15
        first_old_point = max(0,first_diff-distance)
        points_for_corner_finding = np.asfortranarray(np.concatenate((old_points[first_old_point:first_diff], simplified_new_points, old_points[last_diff:last_diff+distance])))
        new_corners = polyline_corners(points_for_corner_finding)
        start = first_diff - first_old_point
        self.corners = np.concatenate((self.corners[:first_diff], new_corners[start:start+simplified_new_points.shape[0]], self.corners[last_diff:]))

    def update_indexes(self, sample2polyline, new_polyline_len):
        old_len = len(self.corners)
        indexes = np.where(self.corners)
        new_polyline_indexes = sample2polyline[indexes]
        self.corners = np.zeros(new_polyline_len, dtype=np.uint8)
        try:
            self.corners[new_polyline_indexes] = 1
        except:
            print(f'{len(self.corners)=} {len(sample2polyline)=} {new_polyline_len=}')

        if self.prev_closest_to_focus_idx != 0: # 0 stays 0
            if self.prev_closest_to_focus_idx == old_len-1: # last stays last
                self.prev_closest_to_focus_idx = len(self.corners)-1
            else:
                try:
                    self.prev_closest_to_focus_idx = sample2polyline[self.prev_closest_to_focus_idx]
                except:
                    # not sure why this is happening; it's happening rarely. it's probably better
                    # to look for the closest point on the polyline at the next mouse move event than
                    # to raise an exception with all of the side effects of that
                    self.prev_closest_to_focus_idx = -1

MIN_ZOOM, MAX_ZOOM = 1, 5

class ZoomTool(Button):
    def on_mouse_down(self, x, y):
        self.start = (x,y)
        da = layout.drawing_area()
        abs_y = y + da.rect[1]
        h = screen.get_height()
        self.max_up_dist = min(.85 * abs_y, h * .3 * (MAX_ZOOM - da.zoom)/(MAX_ZOOM - MIN_ZOOM))
        self.max_down_dist = min(.85 * (h - abs_y), h * .3 * (da.zoom - MIN_ZOOM)/(MAX_ZOOM - MIN_ZOOM))
        self.frame_start = da.xy2frame(x,y)
        self.orig_zoom = da.zoom
        da.set_zoom_center(self.start)
    def on_mouse_up(self, x, y):
        layout.drawing_area().draw()
    def on_mouse_move(self, x, y):
        px, py = self.start
        up = y < py
        da = layout.drawing_area()
        if up and self.max_up_dist == 0:
            new_zoom = MAX_ZOOM
            ratio = 0
        elif not up and self.max_down_dist == 0:
            new_zoom = MIN_ZOOM
            ratio = 1
        else:
            dist = abs(py - y) #math.sqrt(sqdist((x,y), (px,py)))#abs(y - py)
            ratio = min(1, max(0, dist/(self.max_up_dist if up else self.max_down_dist)))
            zoom_change = ratio*((MAX_ZOOM - self.orig_zoom) if up else (self.orig_zoom - MIN_ZOOM))
            if not up:
                zoom_change = -zoom_change
            new_zoom = max(MIN_ZOOM,min(self.orig_zoom + zoom_change, MAX_ZOOM))
        da.set_zoom(new_zoom)

        # we want xy2frame(self.start) to return the same value at the beginnig of the zooming [if possible]
        # we then want xy2frame(iwidth/2, iheight/2) to eventually converge to self.frame_start [if possible]
        # centerx, centery is somewhere between these two "x/yoffset-defining" points
        centerx = (da.iwidth/2)*ratio + px*(1-ratio)
        centery = (da.iheight/2)*ratio + py*(1-ratio)
        framex, framey = self.frame_start
        xoffset = framex/da.xscale - centerx + da.lmargin
        yoffset = framey/da.yscale - centery + da.ymargin
        da.set_xyoffset(xoffset, yoffset)

        da.set_zoom_center(da.frame2xy(*self.frame_start))

class NewDeleteTool(Button):
    def __init__(self, is_new, frame_func, clip_func, layer_func):
        Button.__init__(self)
        self.is_new = is_new
        self.frame_func = frame_func
        self.clip_func = clip_func
        self.layer_func = layer_func

def flood_fill_color_based_on_mask_many_seeds(color_rgba, pen_mask, xs, ys, bucket_color):
    mask_ptr, mask_stride, width, height = greyscale_c_params(pen_mask, is_alpha=False)
    flood_code = 2

    color_ptr, color_stride, color_width, color_height = color_c_params(color_rgba)
    assert color_width == width and color_height == height
    new_color_value = make_color_int(bucket_color)

    rect = np.zeros(4, dtype=np.int32)
    region = arr_base_ptr(rect)

    assert len(xs) == len(ys)
    assert xs.strides == (4,)
    assert ys.strides == (4,)
    x_ptr = arr_base_ptr(xs)
    y_ptr = arr_base_ptr(ys)

    tinylib.flood_fill_color_based_on_mask_many_seeds(color_ptr, mask_ptr, color_stride, mask_stride,
        width, height, region, 0, flood_code, new_color_value, x_ptr, y_ptr, len(xs))
    xmin, ymin, xmax, ymax = rect
    if xmax >= 0 and ymax >= 0:
        return xmin, ymin, xmax-1, ymax-1

def point_line_distance_vectorized(px, py, x1, y1, x2, y2):
    """ Vectorized calculation of distance from multiple points (px, py) to the line segment (x1, y1)-(x2, y2) """
    line_mag = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if line_mag < 1e-8:
        # The line segment is a point
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    
    # Projection of points on the line segment
    u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
    u = np.clip(u, 0, 1)  # Clamping the projection
    
    # Coordinates of the projection points
    ix = x1 + u * (x2 - x1)
    iy = y1 + u * (y2 - y1)
    
    # Distance from points to the projection points
    return np.sqrt((px - ix) ** 2 + (py - iy) ** 2)

def integer_points_near_line_segment(x1, y1, x2, y2, distance):
    """ Vectorized find all integer coordinates within a given distance from the line segment (x1, y1)-(x2, y2) """
    # Determine the bounding box
    xmin = np.floor(min(x1, x2) - distance)
    xmax = np.ceil(max(x1, x2) + distance)
    ymin = np.floor(min(y1, y2) - distance)
    ymax = np.ceil(max(y1, y2) + distance)
    
    # Generate grid of integer points within the bounding box
    x = np.arange(xmin, xmax + 1)
    y = np.arange(ymin, ymax + 1)
    xx, yy = np.meshgrid(x, y)
    
    # Flatten the grids to get coordinates
    px, py = xx.ravel(), yy.ravel()
    
    # Compute distances using vectorized function
    distances = point_line_distance_vectorized(px, py, x1, y1, x2, y2)
    
    # Filter points within the specified distance
    mask = distances <= distance
    result_points = np.vstack((px[mask], py[mask])).T
    
    return result_points.astype(np.int32)

# FIXME: you can't keep paint_bucket_tool - when movies are closed and reopened the objects are recreated
class ChangeColorHistoryItem(HistoryItemBase):
    def __init__(self, paint_bucket_tool, new_color):
        HistoryItemBase.__init__(self)
        self.paint_bucket_tool = paint_bucket_tool
        self.new_color = new_color
    def undo(self):
        old_color = self.paint_bucket_tool.color
        self.paint_bucket_tool.modify_color(self.new_color)
        return ChangeColorHistoryItem(self.paint_bucket_tool, old_color)
    def __str__(self):
        return f'InsertLayerHistoryItem(removing layer {self.layer_pos_before_undo})'

# FIXME: color2tool should be a part of the layout since it's reconstructed when we close/open movies
# last_color should be per movie
class PaintBucketTool(Button):
    color2tool = {}
    last_color = BACKGROUND+(0,)
    @staticmethod
    def choose_last_color():
        set_tool(PaintBucketTool.color2tool[PaintBucketTool.last_color])

    def draw(self,*args):
        if self.color[-1]:
            return
        Button.draw(self,*args)

    def __init__(self,color,change_color=None):
        Button.__init__(self)
        self.color = color
        self.change_color = change_color
        self.px = None
        self.py = None
        self.bboxes = []
        self.pen_mask = None
        self.patching = False
    def fill(self, x, y):
        x, y = layout.drawing_area().xy2frame(x,y)
        x, y = round(x), round(y)

        if self.px is None:
            self.px = x
            self.py = y

        radius = (PAINT_BUCKET_WIDTH//2) * layout.drawing_area().xscale
        points = integer_points_near_line_segment(self.px, self.py, x, y, radius)
        xs = points[:,0]
        ys = points[:,1]
        self.px = x
        self.py = y
        
        color_rgba = surf.pixels3d(movie.edit_curr_frame().surf_by_id('color'))
        bbox = flood_fill_color_based_on_mask_many_seeds(color_rgba, self.pen_mask, xs, ys, self.color)
        if bbox:
            self.bboxes.append(bbox)

            layout.drawing_area().draw_region(bbox)

        PaintBucketTool.last_color = self.color
        
    def on_mouse_down(self, x, y):
        if curr_layer_locked():
            return
        self.patching = ctrl_is_pressed()
        if self.patching:
            NeedleTool().on_mouse_down(x,y)
            return
        self.history_item = HistoryItem('color')
        self.bboxes = []
        self.px = None
        self.py = None
        lines = surf.pixels_alpha(movie.curr_frame().surf_by_id('lines'))
        self.pen_mask = lines == 255

        self.fill(x,y)
    def on_mouse_move(self, x, y):
        if self.patching or curr_layer_locked():
            return
        if self.pen_mask is None: # pen_mask is None has been known to happen in flood_fill_color_based_on_mask_many_seeds...
            self.on_mouse_down(x,y)
        else:
            self.fill(x,y)
    def on_mouse_up(self, x, y):
        if self.patching or curr_layer_locked():
            return
        self.on_mouse_move(x,y)
        if self.bboxes: # we had changes
            inf = 10**9
            minx, miny, maxx, maxy = inf, inf, -inf, -inf
            for iminx, iminy, imaxx, imaxy in self.bboxes:
                minx = min(iminx, minx)
                miny = min(iminy, miny)
                maxx = max(imaxx, maxx)
                maxy = max(imaxy, maxy)
            self.history_item.optimize((minx, miny, maxx, maxy))
            history.append_item(self.history_item)
        self.history_item = None
        self.pen_mask = None
    def modify(self):
        if self.change_color is None:
            return
        widget.setEnabled(False)
        try:
            trace.event('modify-color')
            color = QColorDialog.getColor(QColor(*self.color), options=QColorDialog.DontUseNativeDialog | QColorDialog.ShowAlphaChannel)
            if color.isValid():
                new_color = color.toTuple()
                old_color = self.color
                self.modify_color(new_color)
                history.append_item(ChangeColorHistoryItem(self, old_color))
                return True
        finally:
            widget.setEnabled(True)

    def modify_color(self, new_color):
        tool = PaintBucketTool.color2tool[self.color]
        del PaintBucketTool.color2tool[self.color]
        self.color = new_color 
        PaintBucketTool.color2tool[self.color] = tool
        layout.full_tool.cursor = self.change_color(self.color)
        layout.palette_area().generate_colors_image()

NO_PATH_DIST = 10**6

def skeleton_to_distances(skeleton, x, y):
    dist = np.zeros(skeleton.shape, np.float32)

    sk_ptr, sk_stride, _, _ = greyscale_c_params(skeleton.T, is_alpha=False)
    dist_ptr, dist_stride, width, height = greyscale_c_params(dist.T, expected_xstride=4, is_alpha=False)
    
    maxdist = tinylib.image_dijkstra(sk_ptr, sk_stride, dist_ptr, dist_stride//4, width, height, y, x)

    return dist, maxdist

import colorsys

def tl_skeletonize(mask):
    skeleton = np.zeros(mask.shape,np.uint8)
    mask_ptr, mask_stride, width, height = greyscale_c_params(mask.T, is_alpha=False)
    sk_ptr, sk_stride, _, _ = greyscale_c_params(skeleton.T, is_alpha=False)
    tinylib.skeletonize(mask_ptr, mask_stride, sk_ptr, sk_stride, width, height)
    return skeleton

def fixed_size_region_1d(center, part, full): 
    assert part*2 <= full
    if center < part//2:
        start = 0
    elif center > full - part//2:
        start = full - part
    else:
        start = center - part//2
    return slice(start, start+part)

def fixed_size_image_region(x, y, w, h):
    xs = fixed_size_region_1d(x, w, res.IWIDTH)
    ys = fixed_size_region_1d(y, h, res.IHEIGHT)
    return xs, ys

SK_WIDTH = 350
SK_HEIGHT = 350

# we use a 4-connectivity flood fill so the following 4-pixel pattern is "not a hole":
#
# 0 1
# 1 0
#
# however skeletonize considers this a connected component, so we detect and close such "holes."
# this shouldn't result in closing a 4-connectivity hole like this:
#
# 0 1 0
# 0 0 0
# 1 1 1
#
# since whatever the x values, the middle 0 cannot be closed, only the zeros around it,
# which still leaves a "4-connected hole" that skeletonize will treat as a hole
def close_diagonal_holes(mask):
    diag1 = mask[1:,:-1] & mask[:-1,1:] & ~mask[1:,1:]
    diag2 = mask[:-1,:-1] & mask[1:,1:] & ~mask[:-1,1:]

    # FIXME: handle the full range
    mask[:-1,:-1] |= diag1
    mask[1:,:-1] |= diag2 

def skeletonize_color_based_on_lines(color, lines, x, y):
    pen_mask = lines == 255
    if pen_mask[x,y]:
        return

    flood_code = 2
    flood_mask = np.ascontiguousarray(pen_mask.astype(np.uint8))
    cv2.floodFill(flood_mask, None, seedPoint=(y, x), newVal=flood_code, loDiff=(0, 0, 0, 0), upDiff=(0, 0, 0, 0))
    flood_mask = flood_mask != flood_code
    # note that we should close the diagnoal holes _after_ flood fill (which uses 4-connectivity and doesn't need
    # them closed to work correctly) and not _before_ flood-fill (because it can actually make an 4-connectivity
    # hole into an 8-connectivity hole in some cases and then flood fill will not find it; not a problem after flood-fill
    # since skeletonize uses 8-connectivity)
    close_diagonal_holes(flood_mask)
    flood_mask = ~flood_mask

    #flood_mask = flood_fill(pen_mask.astype(np.byte), (x,y), flood_code) == flood_code
    skx, sky = fixed_size_image_region(x, y, SK_WIDTH, SK_HEIGHT)
    skeleton = tl_skeletonize(np.ascontiguousarray(flood_mask[skx,sky])).astype(np.uint8)

    def dilation(img):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(img, kernel, iterations=1)
    fmb = dilation(dilation(skeleton))

    # Compute distance from each point to the specified center
    d, maxdist = skeleton_to_distances(skeleton, x-skx.start, y-sky.start)

    if maxdist != NO_PATH_DIST:
        d = (d == NO_PATH_DIST)*maxdist + (d != NO_PATH_DIST)*d # replace NO_PATH_DIST with maxdist
    else: # if all the pixels are far from clicked coordinate, make the mask bright instead of dim,
        # otherwise it might look like "the flashlight isn't working"
        #
        # note that this case shouldn't happen because we are highlighting points around the closest
        # point on the skeleton to the clocked coordinate and not around the clicked coordinate itself
        d = np.ones(lines.shape, int)
        maxdist = 10

    outer_d = -dilation(-d)

    maxdist = min(700, maxdist)

    inner = (255,255,255)
    outer = [255-ch for ch in color[x,y]]
    h,s,v = colorsys.rgb_to_hsv(*[o/255. for o in outer])
    s = 1
    v = 1
    outer = [255*o for o in colorsys.hsv_to_rgb(h,s,v)]

    fading_mask = Surface((flood_mask.shape[0], flood_mask.shape[1]))
    fm = surf.pixels3d(fading_mask)
    for ch in range(3):
         fm[skx,sky,ch] = outer[ch]*(1-skeleton) + inner[ch]*skeleton
    surf.pixels_alpha(fading_mask)[skx,sky] = fmb*255*np.maximum(0,pow(1 - outer_d/maxdist, 3))

    return fading_mask, (skeleton, skx, sky)

def splprep(points, weights=None, smoothing=None):
    '''NOTE: scipy.interpolation.splprep expects points (which it calls x) to be transposed relatively to this function, as in
    idim, m = points.shape'''
    m, idim = points.shape
    if weights is None:
        weights = np.ones(m)
    if smoothing is None: 
        smoothing = m - math.sqrt(2*m)
    k = 3
    assert m>k, 'not enough points'

    t = np.zeros(m+k+1)
    c = np.zeros((m+k+1)*idim)
    num_knots = np.zeros(1, np.int32)
    u = np.zeros(m)
    ier = tinylib.fitpack_parcur(arr_base_ptr(points), arr_base_ptr(weights), idim, m, arr_base_ptr(u), k, smoothing,
                                 arr_base_ptr(t), arr_base_ptr(num_knots), arr_base_ptr(c))

    if ier in [1,10]:
        raise Exception(f'splprep - failed to fit a bspline: {ier=} identical consecutive points or zero weights might produce ier=10 {list(points)=} {list(weights)=}')

    n = num_knots[0]
    # c: on succesful exit, this array will contain the coefficients
    #    in the b-spline representation of the spline curve s(u),i.e.
    #       the b-spline coefficients of the spline sj(u) will be given
    #           in c(n*(j-1)+i),i=1,2,...,n-k-1 for j=1,2,...,idim.
    return (t[:n], [c[n*j:n*j+n-k-1] for j in range(idim)], k), u

def splev(x, tck, der=0):
    t, c, k = tck
    try:
        c[0][0]
        parametric = True
    except Exception:
        parametric = False
    if parametric:
        return list(map(lambda c, x=x, t=t, k=k: splev(x, [t, c, k], der=der), c))

    x = np.asarray(x)
    xshape = x.shape
    x = x.ravel()
    y = np.zeros(x.shape, float)
    if der == 0:
        ier = tinylib.fitpack_splev(arr_base_ptr(t), t.shape[0], arr_base_ptr(c), k, arr_base_ptr(x), arr_base_ptr(y), y.shape[0])
    else:
        ier = tinylib.fitpack_splder(arr_base_ptr(t), t.shape[0], arr_base_ptr(c), k, der, arr_base_ptr(x), arr_base_ptr(y), y.shape[0])
    assert ier == 0
    return y.reshape(xshape)

def bspline_interp(points, smoothing=None):
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)

    def dist(i1, i2):
        return math.sqrt((x[i1]-x[i2])**2 + (y[i1]-y[i2])**2)
    curve_length = sum([dist(i, i+1) for i in range(len(x)-1)])

    if smoothing is None:
        smoothing = len(x) / 15
    # scipy.interpolate.splrep works like this:
    #tck, u = splprep([x, y], s=smoothing)
    #ufirst, ulast = u[0], u[-1] # these evaluate to 0, 1
    # our splprep works like this:
    tck, _ = splprep(np.column_stack([x,y]), smoothing=smoothing)
    ufirst, ulast = 0, 1

    step=(ulast-ufirst)/curve_length

    return splev(np.arange(ufirst, ulast+step, step), tck)

HOLE_REGION_W = 40
HOLE_REGION_H = 40

def patch_hole(lines, x, y, skeleton, skx, sky):
    lines_patch = np.ascontiguousarray(lines[skx,sky])
    # pad the lines with 255 if we're near the image boundary, to patch holes near image boundaries

    def add_boundary(arr):
        new = np.zeros((arr.shape[0]+2, arr.shape[1]+2), arr.dtype)
        new[1:-1,1:-1] = arr
        return new

    lines_patch = add_boundary(lines_patch)
    skeleton = add_boundary(skeleton)
    skx = slice(skx.start-1,skx.stop+1)
    sky = slice(sky.start-1,sky.stop+1)

    if skx.start < 0:
        lines_patch[0,:] = 255
    if sky.start < 0:
        lines_patch[:,0] = 255
    if skx.stop > lines.shape[0]:
        lines_patch[-1,:] = 255
    if sky.stop > lines.shape[1]:
        lines_patch[:,-1] = 255

    sk_ptr, sk_stride, _, _ = greyscale_c_params(skeleton.T, is_alpha=False)
    lines_ptr, lines_stride, width, height = greyscale_c_params(lines_patch.T, is_alpha=False)
    
    npoints = 3
    xs = np.zeros(npoints, np.int32)
    ys = np.zeros(npoints, np.int32)

    nextra = 100
    xs1 = np.zeros(nextra, np.int32)
    ys1 = np.zeros(nextra, np.int32)
    n1 = np.array([nextra], np.int32)
    xs2 = np.zeros(nextra, np.int32)
    ys2 = np.zeros(nextra, np.int32)
    n2 = np.array([nextra], np.int32)

    # TODO: if the closest point on the skeleton is near invisible (due to the past distance computation),
    # maybe better to recompute the distances and repaint instead of going ahead and patching?..
    found = tinylib.patch_hole(lines_ptr, lines_stride, sk_ptr, sk_stride, width, height, y-sky.start, x-skx.start,
                               HOLE_REGION_H, HOLE_REGION_W, arr_base_ptr(ys), arr_base_ptr(xs), npoints,
                               arr_base_ptr(ys1), arr_base_ptr(xs1), arr_base_ptr(n1),
                               arr_base_ptr(ys2), arr_base_ptr(xs2), arr_base_ptr(n2))

    if found < 3:
        return False

    n1 = n1[0]
    n2 = n2[0]
    xs1 = xs1[:n1]
    ys1 = ys1[:n1]
    xs2 = xs2[:n2]
    ys2 = ys2[:n2]

    endp1 = xs[0]+skx.start, ys[0]+sky.start
    endp2 = xs[2]+skx.start, ys[2]+sky.start

    if n1 == 0 and n2 == 0: #  just 3 points - create 5 points to fit a curve
        xs = [xs[0], xs[0]*0.9 + xs[1]*0.1, xs[1], xs[1]*0.1 + xs[2]*0.9, xs[2]]
        ys = [ys[0], ys[0]*0.9 + ys[1]*0.1, ys[1], ys[1]*0.1 + ys[2]*0.9, ys[2]]
    else: # we have enough points to not depend on the exact point
        # on the skeleton
        def pad(c, n): # a crude way to add weight to a "lone endpoint",
            # absent this the line fitting can fail to reach it
            eps = 0.0001
            return [c+eps*i for i in range(1 + 9*(n==0))]
        xs = pad(xs[0],n1) + pad(xs[2],n2)
        ys = pad(ys[0],n1) + pad(ys[2],n2)


    oft=0 # at one point it seemed that skeletonization moves the points by .5... currently oft is 0 since it doesn't seem so
    xs = np.concatenate((xs1[::-1]+oft, xs, xs2+oft))
    ys = np.concatenate((ys1[::-1]+oft, ys, ys2+oft))
    lines_patch[:] = 0
    skeleton[:] = 0
    points=[(x+skx.start,y+sky.start) for x,y in zip(xs,ys)]

    def filter_points(px, py):
        start = 0
        end = -1
        for i in range(len(px)):
            if not start and (px[i] - endp1[0])**2 + (py[i] - endp1[1])**2 < 4:
                start = i
                break
        for i in reversed(range(len(px))):
            if end < 0 and (px[i] - endp2[0])**2 + (py[i] - endp2[1])**2 < 4:
                end = i
                break
        return px[start:end], py[start:end]


    path = bspline_interp(points)
    px, py = filter_points(path[0], path[1])
            
    margin = 5
    minx = max(0, math.floor(min(px)) - margin)
    miny = max(0, math.floor(min(py)) - margin)
    maxx = min(res.IWIDTH-1, math.ceil(max(px)) + margin)
    maxy = min(res.IHEIGHT-1, math.ceil(max(py)) + margin)
    history_item = HistoryItem('lines', bbox=(minx, miny, maxx, maxy))

    ptr, ystride, width, height = greyscale_c_params(lines)
    brush = tinylib.brush_init_paint(px[0], py[0], 0, 1, 2.5, 0, 0, 0, 0, ptr, width, height, 4, ystride, 0)
    tinylib.brush_use_max_blending(brush)

    xarr = np.array(px, dtype=float)
    yarr = np.array(py, dtype=float)
    rect = np.zeros(4, dtype=np.int32)
    region = arr_base_ptr(rect)

    tinylib.brush_paint(brush, len(px), arr_base_ptr(xarr), arr_base_ptr(yarr), 0, 0, 1, region)
    tinylib.brush_end_paint(brush, region)

    polyline_length = tinylib.brush_get_polyline_length(brush)
    polyline = np.zeros((polyline_length, 2), dtype=float, order='F')
    polyline_x = polyline[:, 0]
    polyline_y = polyline[:, 1]
    tinylib.brush_get_polyline(brush, polyline_length, arr_base_ptr(polyline_x), arr_base_ptr(polyline_y), 0, 0)

    tinylib.brush_free(brush)

    history_item.editable_pen_line = EditablePenLine(polyline)
    history.append_item(history_item)

    return True

def remove_duplicate_points_ordered(points):
    # Get indices of unique points, preserving first occurrence
    _, indices = np.unique(points, axis=0, return_index=True)
    # Sort indices to maintain original order
    return points[np.sort(indices)]

def filter_points_by_distance(points, threshold):
    """
    Filter 2D points that are further than threshold from their previous point.
    Treats the array as circular (first point compared to last point).

    Parameters:
    points (np.ndarray): Array of 2D points with shape (N, 2)
    threshold (float): Minimum distance threshold

    Returns:
    np.ndarray: Filtered array of points that meet the distance criterion
    """
    if len(points) <= 1:
        return points

    # Create circular differences: each point compared to its previous point
    # For circular array: point[0] compared to point[-1], point[1] to point[0], etc.
    prev_points = np.roll(points, 1, axis=0)  # Shift points: [last, p0, p1, ..., p(n-2)]
    diffs = points - prev_points  # Shape: (N, 2)

    # Calculate distances using L2 norm
    distances = np.linalg.norm(diffs, axis=1)  # Shape: (N,)

    # Find indices where distance >= threshold
    valid_indices = np.where(distances >= threshold)[0]

    return points[valid_indices]


def close_polyline(points):
    # fit a curve thru the endpoints - for that, make them "not the endpoints" but put them
    # in the middle of the curve so that fitpack produces something smooth according to where
    # the curve is going to near the endpoints. (scipy's splprep has a per=1 parameter for 
    # fitting closed curves, which early Tinymation even used to use for closing _and_ smoothing
    # curves early on, but since we now smooth on the fly and don't want to change the curve
    # after it was already painted by a smoothing step, it's an overkill to pull the code implementing
    # per=1 into our fitpack subset)
    orig_points = points

    mid_ind = len(points) // 2
    points = np.concatenate((points[mid_ind:], points[:mid_ind]))

    points = np.column_stack([points[:,0],points[:,1]]) # array layout correction
    tck, u = splprep(points, smoothing=0)#len(points))

    orig_last_ind = len(points[mid_ind:])-1
    orig_first_ind = orig_last_ind+1
    ubegin = u[orig_last_ind]
    uend = u[orig_first_ind]

    sx, sy = orig_points[0,:]
    ex, ey = orig_points[-1,:]
    endpoints_dist = math.sqrt((sx-ex)**2 + (sy-ey)**2)
    step = (uend-ubegin) / (2*endpoints_dist)

    new_points = splev(np.arange(ubegin, uend, step), tck)
    new_points = np.array(list(zip(new_points[0], new_points[1])), dtype=float, order='F')

    # we shouldn't really filter all of the original points, only ones close to new_points, but since the threshold
    # is very small, it doesn't have an effect on anything except the part where the lines connect anyway
    return np.array(filter_points_by_distance(np.concatenate((orig_points, new_points)), 0.1), dtype=float, order='F')

def redraw_line(line, last_item, frame_without_line): 
    affected_bbox = points_bbox(line.points, margin=WIDTH*4)

    minx, miny, maxx, maxy = [round(c) for c in affected_bbox]
    minx, miny = res.clip(minx, miny)
    maxx, maxy = res.clip(maxx, maxy)

    history_item = HistoryItem('lines', bbox=(minx, miny, maxx, maxy))

    lines = movie.edit_curr_frame().surf_by_id('lines')
    if frame_without_line is None:
        frame_without_line = lines.copy()
        last_item.copy_saved_subsurface_into(frame_without_line)

    rgba_array(lines)[minx:maxx,miny:maxy] = rgba_array(frame_without_line)[minx:maxx,miny:maxy]

    pen = TOOLS['pen'].tool
    pen.lines_array = surf.pixels_alpha(lines)
    pen.draw_line(line.points, smoothDist=0, closed=True)
    pen.lines_array = None

    line.frame_without_line = frame_without_line # otherwise, editing the line
    # will expose the original unedited line...
    history_item.editable_pen_line = line
    history.append_item(history_item)

    layout.drawing_area().draw_region(affected_bbox)

def try_to_close_the_last_editable_line(x,y):
    '''if x,y is close to the endpoints of the last editable line and said line isn't already closed, close it'''
    last_item = history.last_item()
    if not last_item or not last_item.editable_pen_line or last_item.editable_pen_line.closed or len(last_item.editable_pen_line.points) < 8:
        return False

    def dist(point):
        px, py = point
        return math.sqrt((px-x)**2 + (py-y)**2)

    hole_radius = math.sqrt(HOLE_REGION_W**2 + HOLE_REGION_H**2)/2

    line = last_item.editable_pen_line
    if dist(line.points[0]) > hole_radius or dist(line.points[-1]) > hole_radius:
        return False

    new_line = EditablePenLine(close_polyline(line.points), closed=True)
    redraw_line(new_line, last_item, line.frame_without_line)
    return True


last_skeleton = None

def ctrl_is_pressed(): return QGuiApplication.queryKeyboardModifiers() & Qt.ControlModifier
def shift_is_pressed(): return QGuiApplication.queryKeyboardModifiers() & Qt.ShiftModifier

class NeedleTool(Button):
    def __init__(self):
        Button.__init__(self)
    def on_mouse_down(self, x, y):
        x, y = layout.drawing_area().xy2frame(x,y)
        x, y = round(x), round(y)

        try_to_patch = ctrl_is_pressed()

        if try_to_close_the_last_editable_line(x,y):
            return

        frame = movie.edit_curr_frame() if try_to_patch else movie.curr_frame()

        color = surf.pixels3d(frame.surf_by_id('color'))
        lines = surf.pixels_alpha(frame.surf_by_id('lines'))
        if x < 0 or y < 0 or x >= color.shape[0] or y >= color.shape[1] or lines[x,y] == 255:
            return

        if try_to_patch:
            # Ctrl pressed - attempt to patch a hole using the previous skeleton (if relevant
            if last_skeleton is not None:
                skeleton, skx, sky = last_skeleton
                if x >= skx.start and x < skx.stop and y >= sky.start and y < sky.stop:
                    found = False
                    if patch_hole(lines, x, y, skeleton, skx, sky):
                        # find a point to compute a new skeleton around. Sometimes x,y itself
                        # is that point and sometimes a neighbor, depending on how the hole was patched.
                        # we want "some" sort of skeleton to give a clear feedback showing that "a hole
                        # was really patched" and the skeleton running into the patch is good feedback.
                        # if we just show no skeleton then if the patch is near invisible it's not clear
                        # what happened. the downside is that we don't know which side of the hole
                        # the new skeleton should be at; we could probably compute several and choose the
                        # largest but seems like too much trouble?..
                        neighbors = [(0,0),(2,0),(-2,0),(0,2),(0,-2),(2,2),(-2,-2),(-2,2),(2,-2)]
                        for ox,oy in neighbors:
                            xi,yi = x+ox,y+oy
                            if xi < 0 or yi < 0 or xi >= color.shape[0] or yi >= color.shape[1]:
                                continue
                            if lines[xi,yi] != 255:
                                break
                                found = True
                        x,y = xi,yi

                        if not found:
                            layout.drawing_area().set_fading_mask(None)
                            return

        fading_mask_and_skeleton = skeletonize_color_based_on_lines(color, lines, x, y)

        if not fading_mask_and_skeleton:
            return
        fading_mask, skeleton = fading_mask_and_skeleton
        fading_mask.set_alpha(255)
        layout.drawing_area().set_fading_mask(fading_mask, skeleton)
        layout.drawing_area().fade_per_frame = 255/(FADING_RATE*15)

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

def scale_rect(rect):
    left, bottom, width, height = rect
    sw = screen.get_width()
    sh = screen.get_height()
    return (round(left*sw), round(bottom*sh), round(width*sw), round(height*sh))

DRAWING_LAYOUT = 1
LAYERS_LAYOUT = 2
ANIMATION_LAYOUT = 3

class Layout:
    def __init__(self):
        self.elems = []
        _, _, self.width, self.height = screen.get_rect()
        self.is_pressed = False
        self.is_playing = False
        self.playing_index = 0
        self.full_tool = TOOLS['pencil']
        self.tool = self.full_tool.tool
        self.focus_elem = None
        self.restore_tool_on_mouse_up = False
        self.mode = ANIMATION_LAYOUT
        self.pressure = 1
        self.event_time = 0
        self.update_roi = None

    def aspect_ratio(self): return self.width/self.height

    def add(self, rect, elem, draw_border=False):
        srect = scale_rect(rect)
        elem.rect = srect
        elem.subsurface = screen.subsurface(srect)
        elem.draw_border = draw_border
        elem.init()
        self.elems.append(elem)

    def freeze(self):
        assert self.elems[0] is self.drawing_area()
        # this is important for vertical_movie_on_horizontal_screen.
        self.elems_event_order = self.elems[1:] + [self.drawing_area()]

    def hidden(self, elem):
        if self.mode == ANIMATION_LAYOUT:
            return False
        if self.mode == LAYERS_LAYOUT:
            return isinstance(elem, TimelineArea) or isinstance(elem, TogglePlaybackButton)
        if self.mode == DRAWING_LAYOUT:
            return isinstance(elem, TimelineArea) or isinstance(elem, LayersArea) or isinstance(elem, TogglePlaybackButton)
        assert False, "Layout: unknown mode"

    def draw(self):
        if self.is_pressed:
            if self.focus_elem is self.drawing_area() and not self.drawing_area().redraw_fading_mask:
                return
            if self.focus_elem is None or not self.focus_elem.redraw:
                return

        screen.fill(UNDRAWABLE)

        if not self.is_playing:
            for elem in self.elems:
                elem.highlight_selection()

        for elem in self.elems:
            if not self.is_playing or isinstance(elem, DrawingArea) or isinstance(elem, TogglePlaybackButton):
                if self.hidden(elem):
                    continue
                try:
                    elem.draw()
                except:
                    import traceback
                    traceback.print_exc()
                    surf.rect(screen, (255,0,0), elem.rect, 3, 3)
                    continue
                if elem.draw_border:
                    surf.rect(screen, OUTLINE, elem.rect, 1, 1)

    def draw_upon_zoom(self):
        cache.lock() # the chance to need to redraw with the same intermediate zoom/pan is low.
        # however the first time we're zooming we better do cache stuff since we might not have had
        # the opportunity to do so beforehand
        self.drawing_area().draw()
        cache.unlock()

        if self.drawing_area().vertical_movie_on_horizontal_screen:
            # since the drawing area is partially covered by the movie list and the timeline, redraw them
            self.movie_list_area().redraw_last()
            self.timeline_area().redraw_last()

    def restore_roi(self, roi):
        da = self.drawing_area()
        if da.vertical_movie_on_horizontal_screen and roi is not None:
            # we might have painted on top of the timeline or the movie area
            x,y,w,h = roi
            movie_list = self.movie_list_area()
            timeline = self.timeline_area()
            if x+w > timeline.rect[0]:
                if y < timeline.rect[3]:
                    timeline.redraw_last()
                if y+h >= movie_list.rect[1]:
                    movie_list.redraw_last()

        if roi is None:
            self.update_roi = (0, 0, 0, 0)
        else:
            self.update_roi = (roi[0]+da.rect[0], roi[1]+da.rect[1], roi[2], roi[3])

    # note that pg seems to miss mousemove events with a Wacom pen when it's not pressed.
    # (not sure if entirely consistently.) no such issue with a regular mouse
    def on_event(self,event):
        if event.type == PLAYBACK_TIMER_EVENT:
            if self.is_playing:
                self.playing_index = (self.playing_index + 1) % len(movie.frames)
            # when zooming/panning, we redraw at the playback rate [instead of per mouse event,
            # which can create a "backlog" where we keep redrawing after the mouse stops moving because we
            # lag after mouse motion.] TODO: do we want to use a similar approach elsewhere?..
            elif self.is_pressed and self.zoom_pan_tool() and self.focus_elem is self.drawing_area():
                self.draw_upon_zoom()

        if event.type == FADING_TIMER_EVENT:
            self.drawing_area().update_fading_mask()

        if event.type == SAVING_TIMER_EVENT:
            movie.frame(movie.pos).save()

        if event.type == HISTORY_TIMER_EVENT:
            self.tool.on_history_timer()

        if event.type not in [MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION]:
            return

        self.pressure = getattr(event, 'pressure', 1)
        self.event_time = getattr(event, 'time', 1)

        if event.type in [MOUSEMOTION, MOUSEBUTTONUP] and not self.is_pressed:
            return # this guards against processing mouse-up with a button pressed which isn't button 0,
            # as well as, hopefully, against various mysterious occurences observed in the wild where
            # we eg are drawing a line even though we aren't actually trying

        x, y = event.pos

        dispatched = False
        for elem in self.elems_event_order:
            left, bottom, width, height = elem.rect
            if x>=left and x<left+width and y>=bottom and y<bottom+height:
                if not self.is_playing or isinstance(elem, TogglePlaybackButton):
                    if self.hidden(elem):
                        continue
                    if elem.hit(x,y):
                        self.subpixel = event.subpixel
                        self._dispatch_event(elem, event, x, y)
                        dispatched = True
                        break

        if not dispatched and self.focus_elem:
            self._dispatch_event(None, event, x, y)
            return

    def _dispatch_event(self, elem, event, x, y):
        if event.type == MOUSEBUTTONDOWN:
            change = tool_change
            self.is_pressed = True
            self.focus_elem = elem
            if self.focus_elem:
                trace.class_context(self.focus_elem)
                self.focus_elem.on_mouse_down(x,y)
            if change == tool_change and self.new_delete_tool():
                self.restore_tool_on_mouse_up = True
        elif event.type == MOUSEBUTTONUP:
            self.is_pressed = False
            if self.restore_tool_on_mouse_up:
                restore_tool()
                self.restore_tool_on_mouse_up = False
                self.focus_elem = None
                return
            if self.focus_elem:
                trace.class_context(self.focus_elem)
                self.focus_elem.on_mouse_up(x,y)
                self.focus_elem = None
        elif event.type == MOUSEMOTION and self.is_pressed:
            if self.focus_elem:
                trace.class_context(self.focus_elem)
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

    def palette_area(self):
        for elem in self.elems:
            if isinstance(elem, PaletteElem):
                return elem
        assert False, 'PaletteElem not found'

    def new_tool(self): return self.new_delete_tool() and self.tool.is_new
    def new_delete_tool(self): return isinstance(self.tool, NewDeleteTool) 
    def zoom_pan_tool(self): return isinstance(self.tool, ZoomTool)
    def needle_tool(self): return isinstance(self.tool, NeedleTool)

    def toggle_playing(self):
        self.is_playing = not self.is_playing
        self.playing_index = 0
            
# assumes either 16:9 or 9:16
def scale_and_fully_preserve_aspect_ratio(w, h, width, height):
    alignw, alignh = (16,9) if w>h  else (9,16)
    if width/height > w/h:
        scaled_width = (round(w*height/h) // alignw) * alignw
        scaled_height = h*scaled_width/w
    else:
        scaled_height = (round(h*width/w) // alignh) * alignh
        scaled_width = w*scaled_height/h
    return round(scaled_width), round(scaled_height)

class DrawingArea(LayoutElemBase):
    def __init__(self, vertical_movie_on_horizontal_screen):
        LayoutElemBase.__init__(self)
        self.vertical_movie_on_horizontal_screen = vertical_movie_on_horizontal_screen
    def init(self):
        self.prescaled_fading_mask = False
        self.fading_mask = None
        self.fading_func = None
        self.fade_per_frame = 0
        self.last_update_time = 0
        self.ymargin = WIDTH * 3
        xmargin = WIDTH * 3
        self.render_surface = None
        self.iwidth = 0
        self.iheight = 0
        self.zoom = 1
        self.zoom_center = (0, 0)
        self.xoffset = 0
        self.yoffset = 0
        self.fading_mask_version = 0
        self.restore_tool_on_mouse_up = False
        self.redraw_fading_mask = False

        left, bottom, width, height = self.rect
        self.iwidth, self.iheight = scale_and_fully_preserve_aspect_ratio(res.IWIDTH, res.IHEIGHT, width - xmargin*2, height - self.ymargin*2)
        assert (self.iwidth / res.IWIDTH) - (self.iheight / res.IHEIGHT) < 1e-9
        xmargin = round((width - self.iwidth)/2)
        self.ymargin = round((height - self.iheight)/2)
        self.set_zoom(self.zoom)

        if self.vertical_movie_on_horizontal_screen:
            self.lmargin = WIDTH*3
            self.rmargin = xmargin*2 - self.lmargin
        else:
            self.lmargin = xmargin
            self.rmargin = xmargin

        w, h = ((self.iwidth+self.lmargin+self.rmargin + self.iheight+self.ymargin*2)//2,)*2
        self.zoom_surface = Surface((w,h ), color=([(a+b)//2 for a,b in zip(MARGIN[:3], BACKGROUND[:3])]))
        rgb = surf.pixels3d(self.zoom_surface)
        alpha = surf.pixels_alpha(self.zoom_surface)
        yv, xv = np.meshgrid(np.arange(h), np.arange(w))
        cx, cy = w/2, h/2
        dist = np.sqrt((xv-cx)**2 + (yv-cy)**2)
        mdist = np.max(dist)
        rgb[dist >= 0.7*mdist] = MARGIN[:3]
        dist = np.minimum(np.maximum(dist, mdist*0.43), mdist*0.7)
        norm_dist = np.maximum(0, dist-mdist*0.43)/(mdist*(0.7-0.43))
        grad = (1 + np.sin(30*norm_dist**1.3))/2
        for i in range(3):
            rgb[:,:,i] = (BACKGROUND[i]*grad +MARGIN[i]*(1-grad))
        alpha[:] = MARGIN[-1]*norm_dist

    def set_xyoffset(self, xoffset, yoffset):
        prevxo, prevyo = self.xoffset, self.yoffset

        # we want xyoffset to be an integer (in the scaled image coordinates; we don't care
        # if it translates to an integer in the frame coordinates.) that's because we scale step-aligned
        # regions of the frame, s.t. their integer coordinates map to integer coordinates in the scaled image.
        # then we display a sub-region of this scaled region, which is not step-aligned; but it has to have
        # integer coordinates so we can just take a sub-region without any warping.

        self.xoffset = int(round(min(max(xoffset, 0), self.iwidth*(self.zoom - 1))))
        self.yoffset = int(round(min(max(yoffset, 0), self.iheight*(self.zoom - 1))))

        zx, zy = self.zoom_center
        self.zoom_center = zx - (self.xoffset - prevxo), zy - (self.yoffset - prevyo)
    def set_zoom(self, zoom, correct=True):
        self.zoom = zoom
        self.xscale = res.IWIDTH/(self.iwidth * self.zoom)
        self.yscale = res.IHEIGHT/(self.iheight * self.zoom)
        if not correct:
            return

        # slightly change the zoom such that not only (0,0) in the original image corresponds to
        # the integer coordinate (0,0) in the scaled output image, but there is some integer step such that
        # (step*M, step*N) for integers M,N in the original image correspond to integer coordinates in the scaled output image.
        # this way we'll be able to scale a step-aligned ROI in the original image and not only the whole image, which
        # is important when quickly redrawing upon pen movement, for example.
        min_step = round(8 / self.xscale)
        try_steps = list(range(min_step,min_step+64))
        orig_image_coordinates_corresponding_to_steps = [i*self.xscale for i in try_steps]
        distances_from_integer_coordinates = [abs(x - round(x)) for x in orig_image_coordinates_corresponding_to_steps]
        min_dist_from_int_coord = min(distances_from_integer_coordinates)
        pos_of_min_dist = distances_from_integer_coordinates.index(min_dist_from_int_coord)
        best_int_step = pos_of_min_dist + try_steps[0]

        #print('zoom',zoom,'best',best_int_step,'dist',min_dist_from_int_coord,'orig_at_step',orig_image_coordinates_corresponding_to_steps[pos_of_min_dist])
        self.zoom_int_step = best_int_step
        self.zoom_int_step_orig = int(round(best_int_step * self.xscale))

        corrected_xscale = round(orig_image_coordinates_corresponding_to_steps[pos_of_min_dist]) / best_int_step
        corrected_zoom = res.IWIDTH / (self.iwidth * corrected_xscale)
        #print('corrected',zoom,corrected_zoom)
        self.set_zoom(corrected_zoom, correct=False)

    def set_zoom_center(self, center): self.zoom_center = center
    def set_zoom_to_film_res(self, center):
        cx, cy = center
        framex, framey = self.xy2frame(cx, cy)

        # in this class, "zoom=1" is zooming to iwidth,iheight; this is zooming to res.IWIDTH,res.IHEIGHT - what would normally be called "1x zoom"
        self.set_zoom(res.IWIDTH / self.iwidth)

        # set xyoffset s.t. the center stays at the same screen location (=we zoom around the center)
        xoffset = framex + self.lmargin - cx
        yoffset = framey + self.ymargin - cy
        self.set_xyoffset(xoffset, yoffset)

        self.set_zoom_center(center)

    def rois(self, just_the_misaligned_frame_roi=False):
        '''step_aligned_frame_roi, scaled_roi_subset, drawing_area_starting_point = da.rois()

        step_aligned_frame_roi is the region in the frame coordinates corresponding to what is shown
        in the drawing area (if we ignore margins, that would be xoffset, yoffset, iwidth, iheight),
        aligned to zoom_int_step_orig. the purpose of this alignment is being able to take step-aligned
        sub-regions of this region (in pen and paint bucket tools), scale them, and get the same pixels
        as we would if we took a sub-region of the scaled image [both correspond to integer coordinates.]
        for this to work, we need zoom to be exactly the value of self.zoom, not just a close approximation,
        and this is only guaranteed if the width & height of the frame region and the scaled region are
        integer multiples of zoom_int_step_orig and zoom_int_step, respectively.

        scaled_roi_subset is the subsurface of the scaled step_aligned_frame_roi that you should take to
        get the pixels to put into the drawing area (meaning, it strips the extra pixels added due to alignment
        and gives you just the pixels in xoffset, yoffset, iwidth, iheight - again, ignoring margins.)

        drawing_area_starting_point is lmargin, ymargin - again, ignoring our drawing at the margins at high zoom;
        this is where scaled_roi_subset should be drawn into da.subsurface.
        '''
        frame_roi = (self.xoffset * self.xscale, self.yoffset * self.yscale, self.iwidth * self.xscale, self.iheight * self.yscale)

        step = 1 if just_the_misaligned_frame_roi else self.zoom_int_step_orig
        def align_down(n): return int((math.floor(n) // step) * step)
        def align_up(n): return int(((math.ceil(n) + step - 1) // step) * step)

        # we fill up the entire drawing area of {l+r}margin + iwidth, 2*ymargin + iheight pixels,
        # unless we want to keep the margins or a part of them empty of pixels - that's when the
        # edges of the image are visible in the shown ROI (which happens when the zoom is small
        # enough or when x/yoffset have the values causing this to be the case)

        frame_roi_left = align_down(max(0, self.xoffset - self.lmargin) * self.xscale)
        frame_roi_bottom = align_down(max(0, self.yoffset - self.ymargin) * self.yscale)
        frame_roi_right = min(res.IWIDTH, align_up((self.xoffset + self.iwidth + self.lmargin + self.rmargin) * self.xscale))
        frame_roi_top = min(res.IHEIGHT, align_up((self.yoffset + self.iheight + self.ymargin*2) * self.yscale))

        def rnd_chk(x):
            r = round(x)
            assert abs(r - x) < 1e-6, f'{r} is not close enough to {x} - should have gotten a value very close to an integer'
            return int(r)

        step_aligned_frame_roi = (frame_roi_left, frame_roi_bottom, frame_roi_right - frame_roi_left, frame_roi_top - frame_roi_bottom)
        if just_the_misaligned_frame_roi:
            return step_aligned_frame_roi # actually it's "aligned" to the step of 1

        # compute target width & height exactly as OpenCV resize would
        scale_target_width = round(step_aligned_frame_roi[2] * (1/self.xscale))
        scale_target_height = round(step_aligned_frame_roi[3] * (1/self.xscale))
        
        scaled_left = max(0, self.xoffset - self.lmargin) - rnd_chk(frame_roi_left/self.xscale)
        scaled_right = max(0, self.yoffset - self.ymargin) - rnd_chk(frame_roi_bottom/self.yscale)
        scaled_width = min(self.iwidth + self.lmargin + self.rmargin, scale_target_width - scaled_left)
        scaled_height = min(self.iheight + self.ymargin*2, scale_target_height - scaled_right)
        scaled_roi_subset = scaled_left, scaled_right, scaled_width, scaled_height 
        
        drawing_area_starting_point = max(0, self.lmargin - self.xoffset), max(0, self.ymargin - self.yoffset)

        return step_aligned_frame_roi, scaled_roi_subset, drawing_area_starting_point
        
    def xy2frame(self, x, y):
        return (x - self.lmargin + self.xoffset)*self.xscale, (y - self.ymargin + self.yoffset)*self.yscale
    def frame2xy(self, framex, framey):
        return framex/self.xscale + self.lmargin - self.xoffset, framey/self.yscale + self.ymargin - self.yoffset
    def roi(self, surface):
        if self.zoom == 1:
            return surface
        return surface.subsurface((self.xoffset, self.yoffset, self.iwidth, self.iheight))
    def scale_and_cache(self, surface, key, get_key=False):
        class ScaledSurface:
            def compute_key(_):
                id2version, comp = key
                return id2version, ('scaled-to-drawing-area', comp, self.zoom, self.xoffset, self.yoffset)
            def compute_value(_):
                step_aligned_frame_roi, scaled_roi_subset, _ = self.rois()
                return scale_image(surface.subsurface(step_aligned_frame_roi), inv_scale=1/self.xscale).subsurface(scaled_roi_subset)
        if get_key:
            return ScaledSurface().compute_key()
        if surface is None:
            return None
        return cache.fetch(ScaledSurface())
    def set_fading_mask(self, fading_mask, skeleton=None, prescaled_fading_mask=False):
        self.fading_mask_version += 1
        cache.update_id('fading-mask', self.fading_mask_version)
        self.fading_mask = fading_mask
        self.prescaled_fading_mask = prescaled_fading_mask
        self.redraw_fading_mask = True
        global last_skeleton
        last_skeleton = skeleton
    def scaled_fading_mask(self):
        if self.prescaled_fading_mask:
            return self.fading_mask
        key = (('fading-mask',self.fading_mask_version),), 'fading-mask'
        m = self.scale_and_cache(self.fading_mask, key)
        m.set_alpha(self.fading_mask.get_alpha())
        return m
    def get_zoom_pan_params(self):
        return self.zoom, self.xoffset, self.yoffset, self.zoom_center, self.xscale, self.yscale, self.zoom_int_step, self.zoom_int_step_orig
    def restore_zoom_pan_params(self, params):
        self.zoom, self.xoffset, self.yoffset, self.zoom_center, self.xscale, self.yscale, self.zoom_int_step, self.zoom_int_step_orig = params
    def reset_zoom_pan_params(self):
        self.set_xyoffset(0, 0)
        self.set_zoom(1)
    def draw(self):
        left, bottom, width, height = self.rect

        if layout.is_playing:
            zoom_params = self.get_zoom_pan_params()
            self.reset_zoom_pan_params()

        if not layout.is_playing:
            self.draw_margin_where_needed((self.lmargin, self.ymargin, self.iwidth, self.iheight), BACKGROUND)

        pos = layout.playing_index if layout.is_playing else movie.pos
        highlight = not layout.is_playing and not movie.curr_layer().locked
        surfaces = []

        step_aligned_frame_roi, scaled_roi_subset, starting_point = self.rois()
        iscale = 1/self.xscale

        surfaces.append(movie.curr_bottom_layers_surface(pos, highlight=highlight, roi=step_aligned_frame_roi, inv_scale=iscale, subset=scaled_roi_subset))
        scaled_curr_layer = None
        if movie.layers[movie.layer_pos].visible:
            scaled_curr_layer = movie.get_thumbnail(pos, transparent_single_layer=movie.layer_pos, roi=step_aligned_frame_roi, inv_scale=iscale).subsurface(scaled_roi_subset)
            surfaces.append(scaled_curr_layer)
        surfaces.append(movie.curr_top_layers_surface(pos, highlight=highlight, roi=step_aligned_frame_roi, inv_scale=iscale, subset=scaled_roi_subset))

        if not layout.is_playing:
            mask = layout.timeline_area().combined_light_table_mask(scaled_curr_layer)
            if mask:
                surfaces.append(mask)

            if self.fading_mask:
                fading = self.scaled_fading_mask()
                if self.prescaled_fading_mask:
                    fading = fading.subsurface(max(0, starting_point[0]), max(0, starting_point[1]), surfaces[0].get_width(), surfaces[0].get_height())
                surfaces.append(fading)

        self.subsurface.blits(surfaces, starting_point)

        margin_area = (starting_point[0], starting_point[1], scaled_roi_subset[2], scaled_roi_subset[3])

        eps = 0.019
        if self.zoom > 1 + eps:
            self.draw_margin_where_needed(margin_area, MARGIN_BLENDED)
            self.draw_zoom_surface()
        else:
            margin_color = UNDRAWABLE if layout.is_playing else MARGIN_BLENDED
            self.draw_margin_where_needed(margin_area, margin_color)

        if layout.is_playing:
            self.restore_zoom_pan_params(zoom_params)

    def draw_margin_where_needed(self, margin_area, margin_color):
        left, bottom, width, height = self.rect
        l,b,w,h = margin_area
        r = l+w
        t = b+h
        surf.box(self.subsurface, (0, 0, l, height), margin_color)
        surf.box(self.subsurface, (r, 0, width-r, height), margin_color)
        surf.box(self.subsurface, (l, 0, r-l, b), margin_color)
        surf.box(self.subsurface, (l, t, r-l, height-t), margin_color)

    def draw_region(self, frame_region):
        trace.stop()
        with trace.start('draw_region'):
            roi = self._draw_region(frame_region)
            layout.restore_roi(roi)

    def _draw_region(self, frame_region):
        xmin, ymin, xmax, ymax = frame_region
        interp_margin = 2 # since we're doing bicubic interpolation, to repaint a region, we need
        # to add a bit of a margin
        xmax += 1 + interp_margin
        ymax += 1 + interp_margin
        xmin -= interp_margin
        ymin -= interp_margin

        # trim region to the currently displayed area [to avoid scaling stuff needlessly]
        (zxmin, zymin, zw, zh), _, _ = self.rois()
        xmin, ymin, xmax, ymax = max(xmin, zxmin), max(ymin, zymin), min(xmax, zxmin+zw), min(ymax, zymin+zh)

        # align the region s.t. it corresponds to integer coordinates in the zoomed image
        step = self.zoom_int_step_orig

        #print('steps', self.zoom_int_step_orig, self.zoom_int_step)
        def align_down(n): return (n // step) * step
        def align_up(n): return ((n + step - 1) // step) * step
        xmin = max(align_down(xmin) - step, 0)
        xmax = min(align_up(xmax) + step, res.IWIDTH)
        ymin = max(align_down(ymin) - step, 0)
        ymax = min(align_up(ymax) + step, res.IHEIGHT)

        x,y = self.frame2xy(xmin, ymin)
        #starting_point = x,y
        #print('starting point',starting_point, 'xyoft', self.xoffset, self.yoffset, 'xyscale', self.xscale, self.yscale)
        sx, sy = round(x),round(y)

        # don't draw the entire scaled area since we have artifacts at the boundaries
        xshrink = 0
        yshrink = 0
        xoft = 0
        yoft = 0
        step = self.zoom_int_step
        if xmin > 0:
            xshrink += step
            sx += step
            xoft = step
        if ymin > 0:
            yshrink += step
            sy += step
            yoft = step
        if xmax < res.IWIDTH:
            xshrink += step
        if ymax < res.IHEIGHT:
            yshrink += step

        if xmax<=xmin or ymax<=ymin:
            return # drawing outside the visible area

        full_step_aligned_frame_roi, full_scaled_roi_subset, full_starting_point = self.rois()
        iscale = 1/self.xscale

        # take the full-region cached surface and take the needed integer sub-region (faster than computing the sub-region by passing roi=src_roi)
        bottom = movie.curr_bottom_layers_surface(movie.pos, highlight=True, roi=full_step_aligned_frame_roi, inv_scale=iscale, subset=full_scaled_roi_subset)
        top = movie.curr_top_layers_surface(movie.pos, highlight=True, roi=full_step_aligned_frame_roi, inv_scale=iscale, subset=full_scaled_roi_subset)
        mask = layout.timeline_area().combined_light_table_mask()

        src_roi = (xmin, ymin, xmax-xmin, ymax-ymin)
        scaled_layer = movie.layers[movie.layer_pos].frame(movie.pos).thumbnail(roi=src_roi, inv_scale=iscale)

        def trim(x,y,w,h,s):
            x = max(x, 0)
            y = max(y, 0)
            right = min(x+w, s.get_width())
            top = min(y+h, s.get_height())
            return x,y,right-x,top-y

        roi = (sx, sy, scaled_layer.get_width() - xshrink, scaled_layer.get_height() - yshrink)
        trimmed_roi = trim(*roi, self.subsurface)
                
        if trimmed_roi[2] <= 0 or trimmed_roi[3] <= 0:
            return # drawing outside the visible area - nothing to repaint

        sub = self.subsurface.subsurface(trimmed_roi)

        other_layers_roi = trim(trimmed_roi[0] - full_starting_point[0], trimmed_roi[1] - full_starting_point[1], trimmed_roi[2], trimmed_roi[3], bottom)
        # printout for the bug where the top surface doesn't contain other_layers_roi
        #print(bottom.get_rect(),top.get_rect(),other_layers_roi)
        sub.blit(bottom.subsurface(other_layers_roi), (0,0)) 
        sub.blit(scaled_layer.subsurface(trim(xoft - (roi[0] - trimmed_roi[0]), yoft - (roi[1] - trimmed_roi[1]), roi[2], roi[3], scaled_layer)), (0,0))
        sub.blit(top.subsurface(other_layers_roi), (0,0)) 
        if mask:
            sub.blit(mask.subsurface(other_layers_roi), (0,0))
        if self.should_draw_zoom_surface():
            self.draw_zoom_surface(trimmed_roi)

        return trimmed_roi

    def should_draw_zoom_surface(self): return self.zoom > 1.015
    def draw_zoom_surface(self, region=None):
        surface = self.subsurface
        sx, sy = 0, 0
        if region is not None:
            surface = surface.subsurface(region)
            sx, sy = -region[0], -region[1]
        start_x = int(self.zoom_center[0] - self.zoom_surface.get_width()/2)
        start_y = int(self.zoom_center[1] - self.zoom_surface.get_height()/2)
        surface.blit(self.zoom_surface, (sx+start_x, sy+start_y))
        end_x = start_x + self.zoom_surface.get_width()
        end_y = start_y + self.zoom_surface.get_height()

        def box(x,y,w,h):
            surf.box(surface, (x,y,w,h), MARGIN)

        box(sx, sy, self.subsurface.get_width(), start_y)
        box(sx, sy+end_y, self.subsurface.get_width(), self.subsurface.get_height())
        box(sx, sy+start_y, start_x, end_y-start_y)
        box(sx+end_x, sy+start_y, self.subsurface.get_width(), end_y-start_y)

    def clear_fading_mask(self):
        global last_skeleton
        last_skeleton = None
        if self.fading_mask is not None:
            self.redraw_fading_mask = True
        self.fading_mask = None
        self.fading_func = None

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
            self.clear_fading_mask()
            return

        if not self.fading_func:
            alpha -= self.fade_per_frame
        else:
            alpha = self.fading_func(alpha, self.fade_per_frame)
        self.fading_mask.set_alpha(max(0,alpha))
        self.redraw_fading_mask = True

    def fix_xy(self,x,y):
        left, bottom, _, _ = self.rect
        return (x-left), (y-bottom)
    def on_mouse_down(self,x,y):
        alt = QGuiApplication.queryKeyboardModifiers() & Qt.AltModifier
        ctrl = QGuiApplication.queryKeyboardModifiers() & Qt.ControlModifier
        if alt:
            set_tool(TOOLS['zoom'])
            layout.restore_tool_on_mouse_up = True
        elif not layout.needle_tool() and not (ctrl and patching_tool_selected()):
            # except upon zooming or patching, we clear the fading mask (if it's purpose is locking
            # we won't get here, and if it's a skeleton, it's invalidated by tool use)
            self.clear_fading_mask()
        trace.class_context(layout.tool)
        layout.tool.on_mouse_down(*self.fix_xy(x,y))
    def on_mouse_up(self,x,y):
        trace.class_context(layout.tool)
        layout.tool.on_mouse_up(*self.fix_xy(x,y))
    def on_mouse_move(self,x,y):
        trace.class_context(layout.tool)
        layout.tool.on_mouse_move(*self.fix_xy(x,y))

class ScrollIndicator:
    def __init__(self, w, h, vertical=False):
        self.vertical = vertical
        self.surface = Surface((w, h))
        scroll_size = (w*2, int(w*2)) if vertical else (int(h*2), h*2)
        self.scroll_left = Surface(scroll_size)
        self.scroll_right = Surface(scroll_size)

        rgb_left = surf.pixels3d(self.scroll_left)
        rgb_right = surf.pixels3d(self.scroll_right)

        y, x = np.meshgrid(np.arange(scroll_size[1]), np.arange(scroll_size[0]))
        s = h
        if vertical:
            x, y = y, x
            s = w
        yhdist = np.abs(y-s)/s
        alpha_left = surf.pixels_alpha(self.scroll_left)
        alpha_right = surf.pixels_alpha(self.scroll_right)
        dist = np.sqrt((y-s)**2 + (x+s/9)**2) # defines the center offset
        dist = np.abs(s/0.78-dist) # defines the ring radius
        dist = np.minimum(dist, s/4.5) # defines the ring width
        mdist = np.max(dist)
        dist /= mdist
        alpha_left[:] = 192*(1-dist)

        dist = np.minimum(dist, 0.5) * 2
        for i in range(3):
            rgb_left[:,:,i] = SELECTED[i]*dist + BACKGROUND[i]*(1-dist)

        if not vertical:
            rgb_right[:] = rgb_left[::-1,:,:]
            alpha_right[:] = alpha_left[::-1,:]
        else:
            rgb_right[:] = rgb_left[:,::-1,:]
            alpha_right[:] = alpha_left[:,::-1]

        self.prev_draw_rect = None
        self.last_dir_change_x = None
        self.last_dir_is_left = None

    def draw(self, surface, px, x, y): # or py, y, x for vertical scroll indicators
        if self.prev_draw_rect is not None:
            try:
                surface.blit(self.surface.subsurface(self.prev_draw_rect), (self.prev_draw_rect[0], self.prev_draw_rect[1]))
            except:
                surface.blit(self.surface, (0, 0))
        y = min(max(y, 0), (surface.get_height() if not self.vertical else surface.get_width())-1)
        if self.last_dir_change_x is None:
            left = px < x
            self.last_dir_change_x = px
        else:
            if abs(x - self.last_dir_change_x) < 10:
                left = self.last_dir_is_left
            else:
                left = self.last_dir_change_x < x
                self.last_dir_change_x = x
        scroll = self.scroll_left if left else self.scroll_right
        if self.vertical:
            starty = x - scroll.get_height()//2
            surface.blit(scroll.subsurface(surface.get_width()-y, 0, scroll.get_width()//2, scroll.get_height()), (0, starty))
            self.prev_draw_rect = (0, starty, scroll.get_width()//2, scroll.get_height())
        else:
            startx = x - scroll.get_width()//2
            surface.blit(scroll.subsurface(0, surface.get_height()-y, scroll.get_width(), scroll.get_height()//2), (startx, 0))
            self.prev_draw_rect = (startx, 0, scroll.get_width(), scroll.get_height()//2)
        self.last_dir_is_left = left

class TimelineArea(LayoutElemBase):
    def _calc_factors(self):
        _, _, width, height = self.rect
        factors = [0.7,0.6,0.5,0.4,0.3,0.2,0.15]
        self.mask_alpha = int(.3*255)
        scale = 1
        mid_scale = 1
        step = 0.5
        mid_width = res.IWIDTH * height / res.IHEIGHT
        def scaled_factors(scale):
            return [min(1, max(0.15, f*scale)) for f in factors]
        def slack(scale):
            total_width = mid_width*mid_scale + 2 * sum([int(mid_width)*f for f in scaled_factors(scale)])
            return width - total_width
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

    def init(self):
        # stuff for drawing the timeline
        self.frame_boundaries = []
        self.eye_boundaries = []
        self.prevx = None

        self._calc_factors()

        eye_icon_size = int(screen.get_width() * 0.15*0.14)
        self.eye_open = scale_image(load_image('light_on.png'), eye_icon_size, best_quality=True)
        self.eye_shut = scale_image(load_image('light_off.png'), eye_icon_size, best_quality=True)

        self.loop_icon = scale_image(load_image('loop.png'), int(screen.get_width()*0.15*0.2), best_quality=True)
        self.arrow_icon = scale_image(load_image('arrow.png'), int(screen.get_width()*0.15*0.2), best_quality=True)

        self.no_hold = scale_image(load_image('no_hold.png'), int(screen.get_width()*0.15*0.11), best_quality=True)
        self.hold_active = scale_image(load_image('hold_yellow.png'), int(screen.get_width()*0.15*0.20), best_quality=True)
        self.hold_inactive = scale_image(load_image('hold_grey.png'), int(screen.get_width()*0.15*0.17), best_quality=True)

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
        
        self.scroll_indicator = ScrollIndicator(self.subsurface.get_width(), self.subsurface.get_height())

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
            color = (brightness,0,0) if pos_dist < 0 else (0,int(brightness*0.5),0)
            transparency = 0.3
            yield (pos, color, transparency)

    def combined_light_table_mask(self, scaled_curr_layer=None):
        # there are 2 kinds of frame positions: those where the frame of the current layer (at movie.layer_pos) is the same
        # as the frame in the current position (at movie.pos) in that layer due to holds, and those where it's not.
        # for the latter kind, we can combine all their masks produced by movie.get_mask together.
        # for the former kind, we don't get_mask to recompute each position's mask when the current layer changes.
        # so instead we do this:
        #   - we combine all the masks containing all the layers *except* the current one
        #   - we additionally combine all the masks containing all the layers *above* the current one
        #   - we then use the first combined mask at the pixels not covered by the current layer's lines/color alpha at movie.pos,
        #     and we use the second combined mask at the pxiels covered by the current layer's lines/color alpha at movie.pos.
        light_table_positions = list(self.light_table_positions())
        curr_frame = movie.curr_frame()
        curr_layer = movie.layers[movie.layer_pos]
        curr_lit = curr_layer.lit and curr_layer.visible
        held_positions = [(pos,c,t) for pos,c,t in light_table_positions if curr_lit and movie.frame(pos) is curr_frame]
        rest_positions = [(pos,c,t) for pos,c,t in light_table_positions if not (curr_lit and movie.frame(pos) is curr_frame)]

        def combine_mask_alphas(mask_alphas):
            if not mask_alphas:
                return None

            mask_params = (MaskAlphaParams * len(mask_alphas))()
            for params, (alpha, rgb) in zip(mask_params, mask_alphas):
                assert alpha.shape == (res.IWIDTH, res.IHEIGHT)
                params.base = arr_base_ptr(alpha)
                params.stride = alpha.strides[1]
                params.rgb = rgb[0] | (rgb[1]<<8) | (rgb[2]<<16);

            combined_mask = Surface((res.IWIDTH, res.IHEIGHT), color=surf.COLOR_UNINIT, alpha=self.mask_alpha)
            mask_base = combined_mask.base_ptr()

            surf.combine_mask_alphas_stat.start()

            @RangeFunc
            def blit_tile(start_y, finish_y):
                tinylib.blit_combined_mask(mask_params, len(mask_params), mask_base, combined_mask.bytes_per_line(), res.IWIDTH, start_y, finish_y)
            tinylib.parallel_for_grain(blit_tile, 0, res.IHEIGHT, 0)

            surf.combine_mask_alphas_stat.stop(res.IHEIGHT*res.IWIDTH*len(mask_alphas))
            
            return combined_mask

        class CachedCombinedMask:
            def __init__(s, light_table_positions, skip_layer=None, lowest_layer_pos=None):
                s.light_table_positions = light_table_positions
                s.skip_layer = skip_layer
                s.lowest_layer_pos = lowest_layer_pos

            def compute_key(s):
                id2version = []
                computation = []
                for pos, color, transparency in s.light_table_positions:
                    i2v, c = movie.get_mask(pos, key=True, lowest_layer_pos=s.lowest_layer_pos, skip_layer=s.skip_layer)
                    id2version += i2v
                    computation.append(('colored-mask',c,color,transparency))
                return tuple(id2version), ('combined-mask', tuple(computation))
                
            def compute_value(s):
                masks = []
                # ATM there's no way to set per-mask transparency from the UI and no code in blit_combined_mask to support it,
                # but both could be added
                for pos, color, transparency in s.light_table_positions:
                    masks.append((movie.get_mask(pos, lowest_layer_pos=s.lowest_layer_pos, skip_layer=s.skip_layer), color))
                return combine_mask_alphas(masks)

        rest_mask = CachedCombinedMask(rest_positions)
        held_mask_outside = CachedCombinedMask(held_positions, skip_layer=movie.layer_pos)
        held_mask_inside = CachedCombinedMask(held_positions, lowest_layer_pos=movie.layer_pos+1)
        da = layout.drawing_area()

        def scaled_key(cached_mask): return da.scale_and_cache(None, cached_mask.compute_key(), get_key=True)
        def scaled(cached_mask):
            k, v = cache.fetch_kv(cached_mask)
            return da.scale_and_cache(v, k)

        class AllPosMask:
            def compute_key(_):
                keys = [scaled_key(m) for m in (rest_mask, held_mask_outside, held_mask_inside)]
                id2vs, comps = zip(*keys)
                id2vs = sum(id2vs,(curr_frame.cache_id_version(),) if held_positions else tuple())
                return id2vs, ('all-pos-mask', tuple(comps))

            def compute_value(_):
                if not held_positions:
                    return scaled(rest_mask)

                held_outside = scaled(held_mask_outside)
                held_inside = scaled(held_mask_inside)
                
                if held_inside or held_outside:
                    step_aligned_frame_roi, scaled_roi_subset, _ = da.rois()
                    if scaled_curr_layer:
                        scaled_layer = scaled_curr_layer
                    else:
                        scaled_layer = movie.get_thumbnail(movie.pos, transparent_single_layer=movie.layer_pos, roi=step_aligned_frame_roi, inv_scale=1/da.xscale).subsurface(scaled_roi_subset)

                    alpha = surf.pixels_alpha(scaled_layer)

                rest = scaled(rest_mask)

                # if we have 0 or 1 valid mask, no blitting needed
                all3 = [held_inside, held_outside, rest]
                valid_count = sum([m is not None for m in all3])
                if valid_count == 0:
                    return None
                if valid_count == 1:
                    for m in all3:
                        if all3 is not None:
                            return m

                for m in all3:
                    if m is not None:
                        w, h = m.get_size()

                # we have 2 or more
                assert held_inside or held_outside

                mask = Surface((w,h), color=surf.COLOR_UNINIT, alpha=self.mask_alpha)
                mask_base = mask.base_ptr()
                layer_base = scaled_layer.base_ptr()

                # pointers are assumed to be non-null and unaliased in ISPC. we allow ourselves to passed
                # aliased pointers as base addresses of unused surfaces...
                def base(s): return s.base_ptr() if s is not None else mask_base
                held_inside_base = base(held_inside)
                held_outside_base = base(held_outside)
                rest_base = base(rest)

                def stride(s): return s.bytes_per_line() if s is not None else 0

                @RangeFunc
                def blit_tile(start_y, finish_y):
                    tmp_row = np.empty(w)
                    tinylib.blit_held_mask(mask_base, mask.bytes_per_line(),
                                           layer_base, scaled_layer.bytes_per_line(),
                                           held_inside_base, stride(held_inside),
                                           held_outside_base, stride(held_outside),
                                           rest_base, stride(rest),
                                           arr_base_ptr(tmp_row), w, start_y, finish_y)
                surf.held_mask_stat.start()
                tinylib.parallel_for_grain(blit_tile, 0, h, 0)
                surf.held_mask_stat.stop(w*h*valid_count)
                return mask
                
        return cache.fetch(AllPosMask())

    def x2frame(self, x):
        for left, right, pos in self.frame_boundaries:
            if x >= left and x <= right:
                return pos
    def draw(self):
        surface = self.scroll_indicator.surface
        surface.fill(UNDRAWABLE)

        left, bottom, width, height = self.rect
        left = 0
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
            surface.blit(scaled, (x, bottom), (0, 0, thumb_width, height))
            list_rect(surface, (x, bottom, thumb_width, height), pos==movie.pos)
            self.frame_boundaries.append((x, x+thumb_width, pos))
            if pos != movie.pos:
                eye = self.eye_open if self.on_light_table.get(pos_dist, False) else self.eye_shut
                eye_x = x + 2 if pos_dist < 0 else x+thumb_width-eye.get_width() - 2
                surface.blit(eye, (eye_x, bottom), eye.get_rect())
                self.eye_boundaries.append((eye_x, bottom, eye_x+eye.get_width(), bottom+eye.get_height(), pos_dist))
            elif len(movie.frames)>1:
                mode = self.loop_icon if self.loop_mode else self.arrow_icon
                mode_x = x + thumb_width - mode.get_width() - 2
                surface.blit(mode, (mode_x, bottom), mode.get_rect())
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

        self.subsurface.blit(surface, (0,0))

    def redraw_last(self):
        surface = self.scroll_indicator.surface
        self.subsurface.blit(surface, (0,0))

    def draw_hold(self):
        left, bottom, width, height = self.rect
        # sort by position for nicer looking occlusion between adjacent icons
        for left, right, pos in sorted(self.frame_boundaries, key=lambda x: x[2]):
            if pos == 0:
                continue # can't toggle hold at frame 0
            if movie.frames[pos].hold:
                hold = self.hold_active if pos == movie.pos else self.hold_inactive
                offset = -hold.get_width() / 2
            elif pos == movie.pos:
                hold = self.no_hold
                offset = hold.get_width() * 0.1
            else:
                continue
            hold_left = left+offset
            hold_bottom = bottom if pos == movie.pos else bottom+height-hold.get_height()*1.1
            self.scroll_indicator.surface.blit(hold, (hold_left, hold_bottom), hold.get_rect())
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

    def fix_x(self,x):
        left, _, _, _ = self.rect
        return x-left
    def on_mouse_down(self,x,y):
        self._on_mouse_down(x,y)
    def _on_mouse_down(self,x,y):
        x = self.fix_x(x)
        self.prevx = None
        if layout.new_delete_tool():
            if self.x2frame(x) == movie.pos:
                layout.tool.frame_func()
            return
        if self.update_on_light_table(x,y):
            return
        if self.update_loop_mode(x,y):
            return
        if self.update_hold(x,y):
            return
        try_set_cursor(finger_cursor[0])
        self.prevx = x
    def on_mouse_up(self,x,y):
        self.on_mouse_move(x,y)
        if self.prevx:
            restore_cursor()
    def on_mouse_move(self,x,y):
        self._on_mouse_move(x,y)
    def _on_mouse_move(self,x,y):
        self.redraw = False
        x = self.fix_x(x)
        if self.prevx is None:
            return
        if layout.new_delete_tool():
            return
        prev_pos = self.x2frame(self.prevx)
        curr_pos = self.x2frame(x)
        self.scroll_indicator.draw(self.subsurface, self.prevx, x, y)
        if prev_pos is None and curr_pos is None:
            self.prevx = x
            return
        if curr_pos is not None and prev_pos is not None:
            pos_dist = prev_pos - curr_pos
        else:
            pos_dist = -1 if x > self.prevx else 1
        self.prevx = x
        if pos_dist != 0:
            self.redraw = True
            if self.loop_mode:
                new_pos = (movie.pos + pos_dist) % len(movie.frames)
            else:
                new_pos = min(max(0, movie.pos + pos_dist), len(movie.frames)-1)
            movie.seek_frame(new_pos)

class LayersArea(LayoutElemBase):
    def init(self):
        left, bottom, width, height = self.rect
        max_height = height / MAX_LAYERS
        max_width = res.IWIDTH * (max_height / res.IHEIGHT)
        self.width = min(max_width, width)
        self.thumbnail_height = int(self.width * res.IHEIGHT / res.IWIDTH)

        self.prevy = None
        icon_height = min(int(screen.get_width() * 0.15*0.14), self.thumbnail_height / 2)
        self.eye_open = scale_image(load_image('eye_open.png'), height=icon_height, best_quality=True)
        self.eye_shut = scale_image(load_image('eye_shut.png'), height=icon_height, best_quality=True)
        self.light_on = scale_image(load_image('light_on.png'), height=icon_height, best_quality=True)
        self.light_off = scale_image(load_image('light_off.png'), height=icon_height, best_quality=True)
        self.locked = scale_image(load_image('locked.png'), height=icon_height*1.2, best_quality=True)
        self.unlocked = scale_image(load_image('unlocked.png'), height=icon_height*1.2, best_quality=True)
        self.eye_boundaries = []
        self.lit_boundaries = []
        self.lock_boundaries = []

        self.scroll_indicator = ScrollIndicator(self.subsurface.get_width(), self.subsurface.get_height(), vertical=True)
    
    def cached_image(self, layer_pos, layer):
        class CachedLayerThumbnail(CachedItem):
            def __init__(s, color=None):
                s.color = color
            def compute_key(s):
                frame = layer.frame(movie.pos) # note that we compute the thumbnail even if the layer is invisible
                return (frame.cache_id_version(),), ('colored-layer-thumbnail', self.width, s.color)
            def compute_value(se):
                if se.color is None:
                    return movie.get_thumbnail(movie.pos, self.width, self.thumbnail_height, transparent_single_layer=layer_pos)
                image = cache.fetch(CachedLayerThumbnail()).copy()
                si = Surface((image.get_width(), image.get_height()), color=BACKGROUND)
                si.blit(image)
                si.blend(se.color + (64,))
                si.set_alpha(128+64)
                return si

        if layer_pos > movie.layer_pos:
            color = LAYERS_ABOVE
        elif layer_pos < movie.layer_pos:
            color = LAYERS_BELOW
        else:
            color = None
        return cache.fetch(CachedLayerThumbnail(color))

    def draw(self):
        surface = self.scroll_indicator.surface
        surface.fill(UNDRAWABLE)

        self.eye_boundaries = []
        self.lit_boundaries = []
        self.lock_boundaries = []

        left, bottom, width, height = self.rect
        blit_bottom = 0

        for layer_pos, layer in reversed(list(enumerate(movie.layers))):
            border = 1 + (layer_pos == movie.layer_pos)*2
            image = self.cached_image(layer_pos, layer)
            image_left = (width - image.get_width())/2
            surf.rect(surface, BACKGROUND, (image_left, blit_bottom, image.get_width(), image.get_height()))
            surface.blit(image, (image_left, blit_bottom), image.get_rect()) 
            list_rect(surface, (image_left, blit_bottom, image.get_width(), image.get_height()), layer_pos == movie.layer_pos)

            max_border = 3
            if len(movie.frames) > 1 and layer.visible and list(layout.timeline_area().light_table_positions()):
                lit = self.light_on if layer.lit else self.light_off
                surface.blit(lit, (width - lit.get_width() - max_border, blit_bottom))
                self.lit_boundaries.append((left + width - lit.get_width() - max_border, bottom, left+width, bottom+lit.get_height(), layer_pos))
               
            eye = self.eye_open if layer.visible else self.eye_shut
            surface.blit(eye, (width - eye.get_width() - max_border, blit_bottom + image.get_height() - eye.get_height() - max_border))
            self.eye_boundaries.append((left + width - eye.get_width() - max_border, bottom + image.get_height() - eye.get_height() - max_border, left+width, bottom+image.get_height(), layer_pos))

            lock = self.locked if layer.locked else self.unlocked
            lock_start = self.thumbnail_height/2 - lock.get_height()/2
            surface.blit(lock, (0, blit_bottom + lock_start))
            self.lock_boundaries.append((left, bottom + lock_start, left+lock.get_width(), bottom + lock_start+lock.get_height(), layer_pos))

            bottom += image.get_height()
            blit_bottom += image.get_height()

        self.subsurface.blit(surface, (0, 0))

    def y2frame(self, y):
        if not self.thumbnail_height:
            return None
        _, bottom, _, _ = self.rect
        return len(movie.layers) - (round(y-bottom) // self.thumbnail_height) - 1

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
        if layout.new_delete_tool():
            if self.y2frame(y) == movie.layer_pos:
                layout.tool.layer_func()
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
            try_set_cursor(finger_cursor[0])
    def on_mouse_up(self,x,y):
        self.on_mouse_move(x,y)
        if self.prevy:
            restore_cursor()
    def on_mouse_move(self,x,y):
        self.redraw = False
        if self.prevy is None:
            return
        if layout.new_delete_tool():
            return
        self.scroll_indicator.draw(self.subsurface, self.prevy-self.rect[1], y-self.rect[1], x-self.rect[0])
        prev_pos = self.y2frame(self.prevy)
        curr_pos = self.y2frame(y)
        if curr_pos is None or curr_pos < 0 or curr_pos >= len(movie.layers):
            return
        self.prevy = y
        pos_dist = curr_pos - prev_pos
        if pos_dist != 0:
            self.redraw = True
            new_pos = min(max(0, movie.layer_pos + pos_dist), len(movie.layers)-1)
            trace.event('seek-layer')
            movie.seek_layer(new_pos)

class ProgressBar:
    def __init__(self, title):
        self.title = title
        self.done = 0
        self.total = 1
        horz_margin = 0.32
        vert_margin = 0.47
        self.inner_rect = scale_rect((horz_margin, vert_margin, 1-horz_margin*2, 1-vert_margin*2))
        left, bottom, width, height = self.inner_rect
        margin = WIDTH
        self.outer_rect = (left-margin, bottom-margin, width+margin*2, height+margin*2)
        self.draw()
        widget.setEnabled(False) # this works better than processEvents(QEventLoop.ExcludeUserInputEvents)
        # which queues the events and then you get them after the progress bar rendering is done
        self.closed = False
    def __del__(self):
        if not self.closed:
            self.close()
    def close(self):
        widget.setEnabled(True)
        self.closed = True
    def on_progress(self, done, total):
        self.done = done
        self.total = total
        self.draw()
    def draw(self):
        surf.rect(screen, UNUSED, self.outer_rect)
        surf.rect(screen, BACKGROUND, self.inner_rect)
        left, bottom, full_width, height = self.inner_rect
        done_width = min(full_width, int(full_width * (self.done/max(1,self.total))))
        surf.rect(screen, PROGRESS, (left, bottom, done_width, height))

        # FIXME
#        text_surface = font.render(self.title, True, UNUSED)
#        pos = ((full_width-text_surface.get_width())/2+left, (height-text_surface.get_height())/2+bottom)
#        screen.blit(text_surface, pos)

        widget.redrawScreen()
        QCoreApplication.processEvents()

def open_movie_with_progress_bar(clipdir):
    trace.event('open-clip')
    progress_bar = ProgressBar('Loading...')
    movie = Movie(clipdir, progress=progress_bar.on_progress)
    progress_bar.close()
    return movie

class MovieList:
    def __init__(self):
        self.reload()
        self.histories = {}
        self.opening = False
    def delete_current_history(self):
        del self.histories[self.clips[self.clip_pos]]
    def reload(self):
        self.clips = []
        self.images = []
        single_image_height = screen.get_height() * MOVIES_Y_SHARE
        for clipdir in get_clip_dirs(sort_by='st_mtime'):
            fulldir = os.path.join(WD, clipdir)
            frame_file = os.path.join(fulldir, CURRENT_FRAME_FILE)
            image = load_image(frame_file) if os.path.exists(frame_file) else new_frame()
            # TODO: avoid reloading images if the file didn't change since the last time
            # we use best quality here since these change rarely and the vertical ones can really look bad otherwise
            self.images.append(scale_image(image, height=single_image_height, best_quality=True))
            self.clips.append(fulldir)
        self.clip_pos = 0#[i for i,clip in enumerate(self.clips) if clip == movie.dir][0]
    def open_clip(self, clip_pos):
        if clip_pos == self.clip_pos:
            return
        global movie
        assert movie.dir == self.clips[self.clip_pos]
        movie.save_before_closing()
        self.clip_pos = clip_pos
        movie = open_movie_with_progress_bar(self.clips[clip_pos])
        self.open_history(clip_pos)
    def open_history(self, clip_pos):
        global history
        history = self.histories.get(self.clips[clip_pos], History())
        movie.restore_viewing_params()
    def save_history(self):
        if self.clips:
            self.histories[self.clips[self.clip_pos]] = history

# 2 questions for a movie list:
# 1. ordering - why by creation date and not by last modification date?
#    since we open the last edited clip upon program start, it seems that you wouldn't mind by-creation ordering very much
#    (you don't often interleave work on several clips created far apart)
#    on the other hand, ordering by last modification has a clear downside manifesting all the time, namely, the order
#    changes all the time which is annoying (you remember where things were and now they all moved.)
# 2. 
class MovieListArea(LayoutElemBase):
    def init(self):
        self.prevx = None
        self.scroll_indicator = ScrollIndicator(self.subsurface.get_width(), self.subsurface.get_height())
        self.pos_pix_share = 0 # between 0 and 1; our leftmost pixel position is sum(thumbnail width)*pos_pix_share
        self.drawn_once = False

        play_icon_size = int(screen.get_width() * 0.15*0.14)
        self.play = scale_image(load_image('play-small.png'), play_icon_size, best_quality=True)
        self.buttons = []
        self.changed_cursor = False
        self.selected_xrange = (-1,-1)
        self.redraw = False

    def thumbnail_widths(self): 
        widths = [im.get_width() for im in movie_list.images]
        accw = []
        sumw = 0
        for w in widths:
            accw.append(sumw)
            sumw += w
        total_width = sum(widths)
        return widths, total_width, accw

    def draw(self):
        surface = self.scroll_indicator.surface
        surface.fill(UNDRAWABLE)

        if not movie_list.opening:
            height = self.rect[-1]
            width = int(res.IWIDTH * height / res.IHEIGHT)
            image = movie.get_thumbnail(movie.pos, width, height, highlight=False) 
            movie_list.images[movie_list.clip_pos] = image # this keeps the image correct when scrolled out of clip_pos
            # (we don't self.reload() upon scrolling so movie_list.images can go stale when the current
            # clip is modified)

        _, _, width, _ = self.rect
        left = 0
        first = True

        widths, total_width, accw = self.thumbnail_widths()

        if not self.drawn_once:
            self.pos_pix_share = max(0, min(self.max_pos_pix_share(total_width), accw[movie_list.clip_pos] / total_width))
            self.drawn_once = True

        leftmost = round(self.pos_pix_share * total_width)

        # TODO: test this code with "not enough movies to fill the area"
        # find the position of the leftmost image we need to draw
        sumw = 0
        for i,w in enumerate(widths):
            sumw += w
            if sumw / total_width >= self.pos_pix_share:
                break

        pos = i
        covered = 0
        self.buttons = []
        for image in movie_list.images[i:]:
            widths_up_to_here = accw[pos]
            startx = covered - (leftmost - widths_up_to_here)
            if startx >= width:
                break
            surface.blit(image, (startx, 0))
            selected = pos == movie_list.clip_pos
            if not selected:
                button_start = (startx + image.get_width()-self.play.get_width(), image.get_height()-self.play.get_height())
                surface.blit(self.play, button_start)
                self.buttons.append((pos,button_start))
            else:
                self.selected_xrange = (startx, startx + image.get_width())
            list_rect(surface, (startx, 0, image.get_width(), image.get_height()), selected)
            leftmost += image.get_width()
            covered += image.get_width()
            pos += 1
        self.subsurface.blit(surface, (0, 0))

    def redraw_last(self):
        surface = self.scroll_indicator.surface
        self.subsurface.blit(surface, (0, 0))

    def x2frame(self, x):
        if not movie_list.images or x is None:
            return None
        left, _, _, _ = self.rect
        return int(round(x-left)) // movie_list.images[0].get_width()
    def button_pressed(self,x,y):
        x -= self.rect[0]
        y -= self.rect[1]
        for pos,(startx,starty) in self.buttons:
            if x>startx and y>starty and x<startx+self.play.get_height():
                movie_list.opening = True
                movie_list.open_clip(pos)
                movie_list.opening = False
                return True
    def in_selected_xrange(self,x):
        x -= self.rect[0]
        startx, endx = self.selected_xrange
        return x>=startx and x<endx
    def on_mouse_down(self,x,y):
        self.prevx = None
        if movie_list.opening:
            return # for some reason the event is delivered again when we run the event loop in ProgressBar;
            # accept()ing the event before calling layout.on_event() doesn't seem to help
        if self.button_pressed(x,y):
            return
        if layout.new_delete_tool():
            # New works anywhere on the movie list area; Delete - only if you hit the selected clip
            if layout.new_tool() or self.in_selected_xrange(x):
                movie_list.opening = True
                layout.tool.clip_func()
                movie_list.opening = False
                self.pos_pix_share = 0
                return
        self.prevx = x
        try_set_cursor(finger_cursor[0])
        self.changed_cursor = True
    def max_pos_pix_share(self,total_width):
        return 1 - self.rect[2]/total_width
    def on_mouse_move(self,x,y):
        if movie_list.opening:
            return
        if self.prevx is None:
            self.prevx = x # this happens eg when a new_delete_tool is used upon mouse down
            # and then the original tool is restored
        if layout.new_delete_tool():
            return

        widths, total_width, accw = self.thumbnail_widths()
        leftmost = (self.pos_pix_share * total_width)

        self.pos_pix_share = max(0, min(self.max_pos_pix_share(total_width), (leftmost - (x - self.prevx)) / total_width))

        self.draw()
        self.scroll_indicator.draw(self.subsurface, self.prevx-self.rect[0], x-self.rect[0], y-self.rect[1])

        self.prevx=x
    def on_mouse_up(self,x,y):
        if self.changed_cursor:
            restore_cursor()
            self.changed_cursor = False
        self.prevx = None
        return

class ToolSelectionButton(LayoutElemBase):
    def __init__(self, tool):
        LayoutElemBase.__init__(self)
        self.tool = tool
    def highlight_selection(self):
        if self.tool is layout.full_tool:
            x,y,w,h = self.rect
            r = SELECTION_CORNER_RADIUS
            exp_rect = [x-r/3,y-r/3,w+r*2/3,h+r*2/3]
            surf.rect(screen, SELECTED, exp_rect)
            surf.rect(screen, UNDRAWABLE, exp_rect, 5, r)
            surf.rect(screen, SELECTED, exp_rect, 2.5, r)
            surf.rect(screen, UNDRAWABLE, exp_rect, 0, r)
    def draw(self):
        self.tool.tool.draw(self.rect,self.tool.cursor[1])
    def hit(self,x,y): return self.tool.tool.hit(x,y,self.rect)
    def on_mouse_down(self,x,y):
        set_tool(self.tool)
        if shift_is_pressed():
            layout.draw()
            widget.redrawScreen()
            if self.tool.tool.modify():
                set_tool(self.tool)
                self.tool.tool.button_surface = None # update from the new icon
        self.redraw = True
    def on_mouse_move(self,x,y): self.redraw = False

class TogglePlaybackButton(Button):
    def __init__(self, play_icon, pause_icon):
        self.play = play_icon
        self.pause = pause_icon
        Button.__init__(self)
        # this is the one button which has a simple, big and round icon, and you don't to be too easy
        # to hit. the others, when we make it necessary to hit the non-transparent part, get annoying -
        # the default behavior is better since imprecise hits select _something_ and then you learn to
        # improve your aim by figuring out what was hit, whereas when nothing is selected you have
        # a harder time learning, apparently
        self.only_hit_non_transparent = True
    def draw(self):
        icon = self.pause if layout.is_playing else self.play
        self.button_surface = None
        Button.draw(self, self.rect, icon)
    def on_mouse_down(self,x,y):
        toggle_playing()
        self.redraw = True
    def on_mouse_move(self,x,y): self.redraw = False

class Tool:
    def __init__(self, tool, cursor, chars):
        self.tool = tool
        self.cursor = cursor
        self.chars = chars

class Movie(MovieData):
    def __init__(self, dir, progress=default_progress_callback):
        iwidth, iheight = (res.IWIDTH, res.IHEIGHT)
        MovieData.__init__(self, dir, progress=progress)

        # load the movie's palette
        palette_file = os.path.join(dir, PALETTE_FILE)
        global palette
        if os.path.exists(palette_file):
            palette = Palette(os.path.join(dir, PALETTE_FILE))
        else: # use the default palette
            palette = Palette(PALETTE_FILE)

        init_layout() # aspect ratio and/or palette might have changed

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

    def frame(self, pos):
        return self.layers[self.layer_pos].frame(pos)

    def get_mask(self, pos, key=False, lowest_layer_pos=None, skip_layer=None):
        # ignore invisible layers
        if lowest_layer_pos is None:
            lowest_layer_pos = 0
        layers = [layer for i,layer in enumerate(self.layers) if layer.visible and i>=lowest_layer_pos and i!=skip_layer]
        # ignore the layers where the frame at the current position is an alias for the frame at the requested position
        # (it's visually noisy to see the same lines colored in different colors all over)
        def lines_lit(layer): return layer.lit and layer.surface_pos(self.pos) != layer.surface_pos(pos)

        class CachedMaskAlpha:
            def compute_key(_):
                frames = [layer.frame(pos) for layer in layers]
                lines = tuple([lines_lit(layer) for layer in layers])
                return tuple([frame.cache_id_version() for frame in frames if not frame.empty()]), ('mask-alpha', lines)
            def compute_value(_):
                layer_params = (LayerParamsForMask * len(layers))()
                for params, layer in zip(layer_params, layers):
                    frame = layer.frame(pos)
                    lines = frame.surf_by_id('lines')
                    color = frame.surf_by_id('color')
                    params.color_base = color.base_ptr()
                    params.color_stride = color.bytes_per_line()
                    params.lines_base = lines.base_ptr()
                    params.lines_stride = lines.bytes_per_line()
                    params.lines_lit = lines_lit(layer)

                alpha = np.ndarray((res.IWIDTH, res.IHEIGHT), strides=(1, res.IWIDTH), dtype=np.uint8)
                alpha_base = arr_base_ptr(alpha)

                surf.get_mask_stat.start()

                @RangeFunc
                def blit_tile(start_y,finish_y):
                    tinylib.blit_layers_mask(layer_params, len(layer_params), alpha_base, alpha.strides[1], alpha.shape[0], start_y, finish_y)
                tinylib.parallel_for_grain(blit_tile, 0, res.IHEIGHT, 0)

                surf.get_mask_stat.stop(res.IWIDTH*res.IHEIGHT*len(layer_params))
                return alpha

        if key:
            return CachedMaskAlpha().compute_key()
        return cache.fetch(CachedMaskAlpha())

    def _visible_layers_id2version(self, layers, pos, include_invisible=False):
        frames = [layer.frame(pos) for layer in layers if layer.visible or include_invisible]
        return tuple([frame.cache_id_version() for frame in frames if not frame.empty()])

    def get_thumbnail(self, pos, width=None, height=None, highlight=True, transparent_single_layer=-1, roi=None, inv_scale=None):
        if roi is None:
            roi = (0, 0, res.IWIDTH, res.IHEIGHT) # the roi is in the original image coordinates, not the thumbnail coordinates
        trans_single = transparent_single_layer >= 0
        layer_pos = self.layer_pos if not trans_single else transparent_single_layer
        def id2version(layers): return self._visible_layers_id2version(layers, pos, include_invisible=trans_single)

        class CachedThumbnail(CachedItem):
            def compute_key(_):
                if trans_single:
                    return id2version([self.layers[layer_pos]]), ('transparent-layer-thumbnail', width, height, roi, inv_scale)
                else:
                    def layer_ids(layers): return tuple([layer.id for layer in layers if not layer.frame(pos).empty()])
                    hl = ('highlight', layer_ids(self.layers[:layer_pos]), layer_ids([self.layers[layer_pos]]), layer_ids(self.layers[layer_pos+1:])) if highlight else 'no-highlight'
                    return id2version(self.layers), ('thumbnail', width, height, roi, hl, inv_scale)
            def compute_value(_):
                h = int(screen.get_height() * 0.15)
                w = int(h * res.IWIDTH / res.IHEIGHT)
                if inv_scale is not None or (w <= width and h <= height):
                    if trans_single:
                        return self.layers[layer_pos].frame(pos).thumbnail(width, height, roi, inv_scale)

                    s = self.curr_bottom_layers_surface(pos, highlight=highlight, width=width, height=height, roi=roi, inv_scale=inv_scale).copy()
                    surfaces = []
                    if self.layers[self.layer_pos].visible:
                        surfaces.append(self.get_thumbnail(pos, width, height, transparent_single_layer=layer_pos, roi=roi, inv_scale=inv_scale))
                    surfaces.append(self.curr_top_layers_surface(pos, highlight=highlight, width=width, height=height, roi=roi, inv_scale=inv_scale))
                    s.blits(surfaces)
                    return s
                else:
                    return scale_image(self.get_thumbnail(pos, w, h, highlight=highlight, transparent_single_layer=transparent_single_layer, roi=roi), width, height)

        return cache.fetch(CachedThumbnail())

    def clear_cache(self):
        layout.drawing_area().clear_fading_mask()

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

    def seek_frame(self,pos):
        trace.event('seek-frame')
        self.seek_frame_and_layer(pos, self.layer_pos)
    def seek_layer(self,layer_pos): self.seek_frame_and_layer(self.pos, layer_pos)

    def next_frame(self): self.seek_frame((self.pos + 1) % len(self.frames))
    def prev_frame(self): self.seek_frame((self.pos - 1) % len(self.frames))

    def next_layer(self): self.seek_layer((self.layer_pos + 1) % len(self.layers))
    def prev_layer(self): self.seek_layer((self.layer_pos - 1) % len(self.layers))

    def insert_frame(self):
        trace.event('insert-frame')
        frame_id = str(uuid.uuid1())
        for layer in self.layers:
            frame = Frame(self.dir, layer.id)
            frame.id = frame_id
            frame.hold = layer is not self.layers[self.layer_pos] # by default, hold the other layers' frames
            layer.frames.insert(self.pos+1, frame)
        self.next_frame()

    def insert_layer(self):
        trace.event('insert-layer')
        frames = [Frame(self.dir, None, frame.id) for frame in self.frames]
        layer = Layer(frames, self.dir)
        self.layers.insert(self.layer_pos+1, layer)
        self.next_layer()

    def reinsert_frame_at_pos(self, pos, removed_frame_data):
        trace.event('insert-frame')
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
        trace.event('insert-layer')
        assert layer_pos >= 0 and layer_pos <= len(self.layers)
        assert len(removed_layer.frames) == len(self.frames)

        self.frame(self.pos).save()
        self.layer_pos = layer_pos

        self.layers.insert(self.layer_pos, removed_layer)
        removed_layer.undelete()

        self.clear_cache()
        self.save_meta()

    def remove_frame(self, at_pos=-1, new_pos=-1):
        trace.event('remove-frame')
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
            # which would be bad since we just called frame.delete() to delete it from the disk

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
        trace.event('remove-layer')
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

    def _set_undrawable_layers_grid(self, s, color, x=0, y=0):
        alpha = surf.pixels3d(s)
        color = np.array(color, dtype=np.uint8)
        alpha[x::WIDTH*3, y::WIDTH*3, :] = color
        alpha[x+1::WIDTH*3, y::WIDTH*3, :] = color
        alpha[x::WIDTH*3, y+1::WIDTH*3, :] = color
        alpha[x+1::WIDTH*3, y+1::WIDTH*3, :] = color

    def curr_bottom_layers_surface(self, pos, highlight, width=None, height=None, roi=None, inv_scale=None, subset=None, transparent=False):
        if not width and not inv_scale: width=res.IWIDTH
        if not height and not inv_scale: height=res.IHEIGHT
        if not roi: roi=(0, 0, res.IWIDTH, res.IHEIGHT)

        class CachedBottomLayers:
            def compute_key(_):
                if transparent:
                    kind = 'transparent-layers'
                else:
                    kind = 'blit-bottom-layers' if not highlight else 'bottom-layers-highlighted'
                return self._visible_layers_id2version(self.layers[:self.layer_pos], pos), (kind, width, height, roi, inv_scale, subset)
            def compute_value(_):
                if transparent:
                    return self._blit_layers(self.layers[:self.layer_pos], pos, transparent=True, width=width, height=height, roi=roi, inv_scale=inv_scale)

                swidth, sheight = scaled_image_size(res.IWIDTH, res.IHEIGHT, width, height, inv_scale)
                if swidth*sheight < (res.IWIDTH*res.IHEIGHT)*0.1:
                    # for small thumbnails, faster to blit scaled layers since we have cached scaled thumbnails per layer which remain valid
                    # while scrolling the timeline or area list
                    layers = self._blit_layers(self.layers[:self.layer_pos], pos, transparent=True, width=width, height=height, roi=roi, inv_scale=inv_scale)
                else:
                    # for large surfaces / drawing area, faster to blit in full resolution first and cache the result, so that when the zoom
                    # changes, we only need to scale the ROI.
                    layers = self.curr_bottom_layers_surface(pos, highlight, transparent=True)
                    layers = scale_image(layers.subsurface(roi), width, height, inv_scale)

                layers = layers.subsurface(subset) if subset is not None else layers

                s = Surface((layers.get_width(), layers.get_height()), color=BACKGROUND)
                if self.layer_pos == 0:
                    return s
                if not highlight:
                    s.blit(layers)
                    return s

                layers.blend(LAYERS_BELOW + (128,))
                self._set_undrawable_layers_grid(layers, (0,0,255))
                s.blit(layers)
                return s

        return cache.fetch(CachedBottomLayers())

    def curr_top_layers_surface(self, pos, highlight, width=None, height=None, roi=None, inv_scale=None, subset=None, transparent=False):
        if not width and not inv_scale: width=res.IWIDTH
        if not height and not inv_scale: height=res.IHEIGHT
        if not roi: roi=(0, 0, res.IWIDTH, res.IHEIGHT)

        class CachedTopLayers:
            def compute_key(_):
                if transparent:
                    kind = 'transparent-layers'
                else:
                    kind = 'blit-top-layers' if not highlight else 'top-layers-highlighted'
                return self._visible_layers_id2version(self.layers[self.layer_pos+1:], pos), (kind, width, height, roi, inv_scale, subset)
            def compute_value(_):
                if transparent:
                    return self._blit_layers(self.layers[self.layer_pos+1:], pos, transparent=True, width=width, height=height, roi=roi, inv_scale=inv_scale)

                swidth, sheight = scaled_image_size(res.IWIDTH, res.IHEIGHT, width, height, inv_scale)
                if swidth*sheight < (res.IWIDTH*res.IHEIGHT)*0.1:
                    layers = self._blit_layers(self.layers[self.layer_pos+1:], pos, transparent=True, width=width, height=height, roi=roi, inv_scale=inv_scale)
                else:
                    layers = self.curr_top_layers_surface(pos, highlight, transparent=True)
                    layers = scale_image(layers.subsurface(roi), width, height, inv_scale)

                layers = layers.subsurface(subset) if subset is not None else layers

                if not highlight or self.layer_pos == len(self.layers)-1:
                    return layers

                layers.blend(LAYERS_ABOVE + (128,))
                layers.set_alpha(192)
                self._set_undrawable_layers_grid(layers, (255,0,0), x=WIDTH*3//2, y=WIDTH**3//2)
                return layers

        return cache.fetch(CachedTopLayers())

    def render_and_save_current_frame(self):
        surf.save(self._blit_layers(self.layers, self.pos), os.path.join(self.dir, CURRENT_FRAME_FILE))

    def garbage_collect_layer_dirs(self):
        # we don't remove deleted layers from the disk when they're deleted since if there are a lot
        # of frames, this could be slow. those deleted layers not later un-deleted by the removal ops being undone
        # will be garbage-collected here
        for f in os.listdir(self.dir):
            full = os.path.join(self.dir, f)
            if f.endswith('-deleted') and f.startswith('layer-') and os.path.isdir(full):
                shutil.rmtree(full)

    def _save_without_export_or_closing(self):
        '''to export, call save_and_export(); to close, call save_before_closing()'''
        self.frame(self.pos).save()
        self.save_meta()

        for layer in self.layers:
            for frame in layer.frames:
                frame.wait_for_compression_to_finish()

    def save_and_export(self):
        self._save_without_export_or_closing()

        # remove old exported files so we don't have stale ones lying around that don't correspond to a valid frame
        # (or a stale MP4/GIF for a now-single-frame clip or vice versa);
        # also, we use the pngs specifically for getting the status of the export progress...
        for f in os.listdir(self.dir):
            if is_exported_png(f):
                os.unlink(os.path.join(self.dir, f))
        for f in self.export_paths_outside_clipdir():
            if os.path.exists(f):
                os.unlink(f)

        self._export_with_progress_bar()

    def _export_with_progress_bar(self):
        future = executor.submit(export, self)

        progress_bar = ProgressBar('Exporting...')
        progress_status = ExportProgressStatus(self.dir, len(self.frames))

        while not future.done():
            progress_status.update([self.dir])
            progress_bar.on_progress(progress_status.done, progress_status.total)
            time.sleep(0.3)

        progress_bar.close()

    def save_before_closing(self):
        self._save_without_export_or_closing()

        movie_list.save_history()
        global history
        history = History()

        self.render_and_save_current_frame()
        palette.save(os.path.join(self.dir, PALETTE_FILE))

        self.garbage_collect_layer_dirs()
        self.mark_as_garbage_in_cache()

    def mark_as_garbage_in_cache(self):
        for layer in self.layers:
            for frame in layer.frames:
                frame.mark_as_garbage_in_cache()

    def fit_to_resolution(self):
        for layer in self.layers:
            for frame in layer.frames:
                frame.fit_to_resolution()

    def delete(self): self.rename(movie.dir + '-deleted')
    def rename(self, new_path):
        for f in self.export_paths_outside_clipdir():
            if os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        os.rename(self.dir, new_path)
        self.dir = new_path

class InsertFrameHistoryItem(HistoryItemBase):
    def __init__(self):
        HistoryItemBase.__init__(self)
    def undo(self):
        # normally remove_frame brings you to the next frame after the one you removed.
        # but when undoing insert_frame, we bring you to the previous frame after the one
        # you removed - it's the one where you inserted the frame we're now removing to undo
        # the insert, so this is where we should go to bring you back in time.
        removed_frame_data = movie.remove_frame(at_pos=self.pos_before_undo, new_pos=max(0, self.pos_before_undo-1))
        return RemoveFrameHistoryItem(self.pos_before_undo, removed_frame_data)
    def __str__(self):
        return f'InsertFrameHistoryItem(removing at pos {self.pos_before_undo})'

class RemoveFrameHistoryItem(HistoryItemBase):
    def __init__(self, pos, removed_frame_data):
        HistoryItemBase.__init__(self, restore_pos_before_undo=False)
        self.pos = pos
        self.removed_frame_data = removed_frame_data
    def undo(self):
        movie.reinsert_frame_at_pos(self.pos, self.removed_frame_data)
        return InsertFrameHistoryItem()
    def __str__(self):
        return f'RemoveFrameHistoryItem(inserting at pos {self.pos})'
    def byte_size(self):
        frames, holds = self.removed_frame_data
        return sum([f.size() for f in frames])

class InsertLayerHistoryItem(HistoryItemBase):
    def __init__(self):
        HistoryItemBase.__init__(self)
    def undo(self):
        removed_layer = movie.remove_layer(at_pos=self.layer_pos_before_undo, new_pos=max(0, self.layer_pos_before_undo-1))
        return RemoveLayerHistoryItem(self.layer_pos_before_undo, removed_layer)
    def __str__(self):
        return f'InsertLayerHistoryItem(removing layer {self.layer_pos_before_undo})'

class RemoveLayerHistoryItem(HistoryItemBase):
    def __init__(self, layer_pos, removed_layer):
        HistoryItemBase.__init__(self, restore_pos_before_undo=False)
        self.layer_pos = layer_pos
        self.removed_layer = removed_layer
    def undo(self):
        movie.reinsert_layer_at_pos(self.layer_pos, self.removed_layer)
        return InsertLayerHistoryItem()
    def __str__(self):
        return f'RemoveLayerHistoryItem(inserting layer {self.layer_pos})'
    def byte_size(self):
        return sum([f.size() for f in self.removed_layer.frames])

class ToggleHoldHistoryItem(HistoryItemBase):
    def __init__(self): HistoryItemBase.__init__(self)
    def undo(self):
        movie.toggle_hold()
        return self
    def __str__(self):
        return f'ToggleHoldHistoryItem(toggling hold at frame {self.pos_before_undo} layer {self.layer_pos_before_undo})'

class ToggleHistoryItem(HistoryItemBase):
    def __init__(self, toggle_func):
        HistoryItemBase.__init__(self) # ATM the toggles we use require to seek
        # to the original movie position before undoing - could make this more parameteric if needed
        self.toggle_func = toggle_func
    def undo(self):
        self.toggle_func()
        return self
    def __str__(self):
        return f'ToggleHistoryItem({self.toggle_func.__qualname__})'

def insert_frame():
    movie.insert_frame()
    history.append_item(InsertFrameHistoryItem())

def insert_layer():
    movie.insert_layer()
    history.append_item(InsertLayerHistoryItem())

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
    movie.next_frame()

def prev_frame():
    if movie.pos <= 0 and not layout.timeline_area().loop_mode:
        return
    movie.prev_frame()

def next_layer():
    movie.next_layer()

def prev_layer():
    movie.prev_layer()

def insert_clip():
    global movie
    movie.save_before_closing()
    movie = Movie(new_movie_clip_dir())
    movie.render_and_save_current_frame() # write out CURRENT_FRAME_FILE for MovieListArea.reload...
    movie_list.reload()

def remove_clip():
    if len(movie_list.clips) <= 1:
        return # we don't remove the last clip - if we did we'd need to create a blank one,
        # which is a bit confusing. [we can't remove the last frame in a timeline, either]
    global movie
    movie.save_before_closing()
    movie.delete()
    movie_list.delete_current_history()
    movie_list.reload()

    new_clip_pos = 0
    movie = open_movie_with_progress_bar(movie_list.clips[new_clip_pos])
    movie_list.clip_pos = new_clip_pos
    movie_list.open_history(new_clip_pos)

class RenameDialog(QDialog):
    def __init__(self, initial_text="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rename clip")
        
        # Create main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Create line edit widget
        self.edit_box = QLineEdit(self)
        self.edit_box.setText(initial_text)
        self.edit_box.selectAll()  # Select all text initially
        
        # Create buttons
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("OK", self)
        self.cancel_button = QPushButton("Cancel", self)
        
        # Connect buttons to slots
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        # Add buttons to button layout
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        # Add widgets to main layout
        main_layout.addWidget(self.edit_box)
        main_layout.addLayout(button_layout)

    def get_text(self):
        return self.edit_box.text()

import pathvalidate

def rename_clip():
    trace.event('rename-clip')
    curr_name = os.path.basename(movie.dir)
    done = False
    widget.setEnabled(False)
    try:
        while not done:
            dialog = RenameDialog(curr_name, parent=widget)
            if dialog.exec() == QDialog.Accepted:
                try:
                    pathvalidate.validate_filename(dialog.get_text())
                except Exception as e:
                    msg_box = QMessageBox()
                    msg_box.setWindowTitle('Invalid file name!')
                    msg_box.setText(f"'{dialog.get_text()}' is not a valid file name\n\n{e}")
                    msg_box.setStandardButtons(QMessageBox.Ok)
                    msg_box.exec()
                    continue

                movie.rename(os.path.join(os.path.dirname(movie.dir), dialog.get_text()))
                movie_list.clips[movie_list.clip_pos] = movie.dir
            done = True
    finally:
        widget.setEnabled(True)

def toggle_playing(): layout.toggle_playing()

def toggle_loop_mode():
    timeline_area = layout.timeline_area()
    timeline_area.loop_mode = not timeline_area.loop_mode

def toggle_frame_hold():
    if movie.pos != 0 and not curr_layer_locked():
        movie.toggle_hold()
        history.append_item(ToggleHoldHistoryItem())

def toggle_layer_lock():
    layer = movie.curr_layer()
    layer.toggle_locked()
    history.append_item(ToggleHistoryItem(layer.toggle_locked))

def zoom_to_film_res():
    pos = QCursor.pos()
    x, y = pos.x(), pos.y()
    da = layout.drawing_area()
    left, bottom, width, height = da.rect
    da.set_zoom_to_film_res((x-left,y-bottom))

TOOLS = {
    'zoom': Tool(ZoomTool(), zoom_cursor, 'zZ'),
    'pen': Tool(PenTool(width=2.5, zoom_changes_pixel_width=False), pen_cursor, 'bB'),
    'pencil': Tool(PenTool(soft=True, width=4, zoom_changes_pixel_width=False), pencil_cursor, 'sS'),
    # TODO: do a reasonable tool UI for this
    'rgb4lines': Tool(PenTool(soft=True, width=20, zoom_changes_pixel_width=True, rgb=(255,80,192)), pencil_cursor, 'qQ'),
    'eraser': Tool(PenTool(eraser=True, soft=True, width=4, zoom_changes_pixel_width=False), eraser_cursor, 'wW'),
    'eraser-medium': Tool(PenTool(eraser=True, soft=True, width=MEDIUM_ERASER_WIDTH), eraser_medium_cursor, 'eE'),
    'eraser-big': Tool(PenTool(eraser=True, soft=True, width=BIG_ERASER_WIDTH), eraser_big_cursor, 'rR'),
    'tweezers': Tool(TweezersTool(), tweezers_cursor, 'mM'),
    'needle': Tool(NeedleTool(), flashlight_cursor, 'nN'), # needle
    # insert/remove frame are both a "tool" (with a special cursor) and a "function."
    # meaning, when it's used thru a keyboard shortcut, a frame is inserted/removed
    # without any more ceremony. but when it's used thru a button, the user needs to
    # actually press on the current image in the timeline to remove/insert. this,
    # to avoid accidental removes/inserts thru misclicks and a resulting confusion
    # (a changing cursor is more obviously "I clicked a wrong button, I should click
    # a different one" than inserting/removing a frame where you need to undo but to
    # do that, you need to understand what just happened)
    'insert-frame': Tool(NewDeleteTool(True, insert_frame, insert_clip, insert_layer), blank_page_cursor, ''),
    'remove-frame': Tool(NewDeleteTool(False, remove_frame, remove_clip, remove_layer), garbage_bin_cursor, ''),
}

def help_screen():
    import help
    images = {}
    for name, angle, height in help.icons:
        images[name.replace('-','_')] = scale_image(surf.rotate(load_image(f"{name}.png"), angle), height=height, best_quality=True).qimage()
    help.HelpDialog(images).exec()

FUNCTIONS = {
    'insert-frame': (insert_frame, '=+'),
    'remove-frame': (remove_frame, '-_'),
    'next-frame': (next_frame, ['>','.',Qt.Key_Right]),
    'prev-frame': (prev_frame, ['<',',',Qt.Key_Left]),
    'next-layer': (next_layer, [Qt.Key_Up]),
    'prev-layer': (prev_layer, [Qt.Key_Down]),
    'toggle-playing': (toggle_playing, [Qt.Key_Enter, Qt.Key_Return]),
    'toggle-frame-hold': (toggle_frame_hold, 'hH'),
    'toggle-layer-lock': (toggle_layer_lock, 'lL'),
    'zoom-to-film-res': (zoom_to_film_res, '1'),
    'last-paint-bucket': (PaintBucketTool.choose_last_color, 'kKcC'),
}

tool_change = 0
prev_tool = None
def set_tool(tool):
    global prev_tool
    global tool_change
    prev = layout.full_tool
    layout.tool = tool.tool
    layout.full_tool = tool
    if not isinstance(prev.tool, NewDeleteTool):
        prev_tool = prev
    if tool.cursor:
        try_set_cursor(tool.cursor[0])
    tool_change += 1

def restore_tool():
    set_tool(prev_tool)

def color_image(s, rgba):
    sc = s.copy()
    pixels = surf.pixels3d(sc)

    alphas = surf.pixels_alpha(sc)
    alphas[:] = np.minimum(alphas[:], ((255 - pixels[:,:,0]).astype(np.int32) + rgba[-1]))

    for ch in range(3):
        pixels[:,:,ch] = (pixels[:,:,ch].astype(int)*rgba[ch])//255
    return sc

import random

PALETTE_ROWS = 11
PALETTE_COLUMNS = 3

class Palette:
    splashes = [load_image(f) for f in ['splash-%d.png'%n for n in range(PALETTE_COLUMNS*PALETTE_ROWS)]]
    random.seed(time.time())
    random.shuffle(splashes)

    def __init__(self, filename, rows=PALETTE_ROWS, columns=PALETTE_COLUMNS):
        s = load_image(filename)
        color_hist = {}
        first_color_hit = {}
        for x in range(s.get_width()):
            for y in range(s.get_height()):
                color = tuple(s.get_at((x,y)))
                if color not in first_color_hit:
                    first_color_hit[color] = (x / (s.get_width()/3))*s.get_height() + y
                color_hist[color] = color_hist.get(color,0) + 1

        colors = [[None for col in range(columns)] for row in range(rows)]
        color2popularity = dict(list(reversed(sorted(list(color_hist.items()), key=lambda x: x[1])))[:rows*columns])
        hit2color = [(first_hit, color) for color, first_hit in sorted(list(first_color_hit.items()), key=lambda x: x[1])]

        row = 0
        col = 0
        for hit, color in hit2color:
            if color in color2popularity:
                colors[rows-row-1][col] = color
                row+=1
                if row == rows:
                    col += 1
                    row = 0

        self.bg_color = BACKGROUND+(0,)

        self.rows = rows
        self.columns = columns
        self.colors = colors

        self.init_cursors()

    def save(self, filename):
        pix = 20
        s = Surface((self.columns*pix, self.rows*pix))
        rgb = surf.pixels3d(s)
        alpha = surf.pixels_alpha(s)
        for row in range(self.rows):
            for col in range(self.columns):
                color = self.colors[self.rows-row-1][col]
                rgb[col*pix:(col+1)*pix,row*pix:(row+1)*pix] = color[:3]
                alpha[col*pix:(col+1)*pix,row*pix:(row+1)*pix] = color[-1]
        surf.save(s, filename)

    def bucket(self,color):
        radius = PAINT_BUCKET_WIDTH//2
        return add_circle(color_image(paint_bucket_cursor[0], color), radius)[0]

    def init_cursors(self):
        radius = PAINT_BUCKET_WIDTH//2
        self.cursors = [[None for col in range(self.columns)] for row in range(self.rows)]
        for row in range(self.rows):
            for col in range(self.columns):
                self.change_color(row, col, self.colors[row][col])

        sc = self.bucket(self.bg_color)
        cursor = (surf2cursor(sc, radius,sc.get_height()-radius-1), color_image(paint_bucket_cursor[1], self.bg_color))
        self.bg_cursor = (cursor[0], scale_image(load_image('water-tool.png'), cursor[1].get_width(), best_quality=True))
        
    def change_color_func(self, r, c): return lambda color: self.change_color(r, c, color)
    def change_color(self, row, col, color):
        radius = PAINT_BUCKET_WIDTH//2
        self.colors[row][col] = color
        sc = self.bucket(color)
        self.cursors[row][col] = (surf2cursor(sc, radius,sc.get_height()-radius-1), color_image(self.splashes[row*self.columns+col], color))
        return self.cursors[row][col]


class PaletteElem(LayoutElemBase):
    row_col_perm = [(row,col) for row in range(PALETTE_ROWS) for col in range(PALETTE_COLUMNS)]
    random.shuffle(row_col_perm)

    def init(self):
        self.generate_colors_image()
        
    def generate_colors_image(self):
        l,b,w,h = self.rect
        col_width = w/palette.columns
        row_height = h/palette.rows

        self.colors_image = Surface((w,h))
        for row,col in self.row_col_perm:
            scale = 1.2 if row != 0 and row != palette.rows-1 else 1.1
            img = palette.cursors[palette.rows-row-1][col][1]
            w, h = scale_and_preserve_aspect_ratio(img.get_width(), img.get_height(), col_width*scale, row_height*scale)
            s = scale_image(img, w, h, best_quality=True)
            offsetx = (col_width*scale - w) / 2
            offsety = (row_height*scale - h) / 2
            if row == 0:
                offsety += row_height*(scale-1)/2
            elif row == palette.rows-1:
                offsety -= row_height*(scale-1)/2
            self.colors_image.blit(s, (offsetx + (col - (scale-1)/2)*col_width, offsety + (row - (scale-1)/2)*row_height))

    def draw(self):
        l,b,_,_ = self.rect
        screen.blit(self.colors_image, (l,b))

def get_clip_dirs(sort_by): # sort_by is a stat struct attribute (st_something)
    '''returns the clip directories sorted by last modification time (latest first)'''
    wdfiles = os.listdir(WD)
    clipdirs = {}
    for d in wdfiles:
        try:
            if d.endswith('-deleted'):
                continue
            frame_order_file = os.path.join(os.path.join(WD, d), CURRENT_FRAME_FILE)
            s = os.stat(frame_order_file)
            clipdirs[d] = getattr(s, sort_by)
        except:
            continue

    return list(reversed(sorted(clipdirs.keys(), key=lambda d: clipdirs[d])))

MAX_LAYERS = 8

layout = None
palette = Palette(PALETTE_FILE)

def init_layout():
    global layout
    global MOVIES_Y_SHARE

    vertical_movie_on_horizontal_screen = res.IWIDTH < res.IHEIGHT and screen.get_width() > 1.5*screen.get_height()

    TIMELINE_Y_SHARE = 0.15
    TOOLBAR_X_SHARE = 0.15
    LAYERS_Y_SHARE = 1-TIMELINE_Y_SHARE
    LAYERS_X_SHARE = LAYERS_Y_SHARE / MAX_LAYERS
    
    # needs to be the same in all layouts or we need to adjust thumbnails in movie_list.images
    MOVIES_Y_SHARE = TOOLBAR_X_SHARE + LAYERS_X_SHARE - TIMELINE_Y_SHARE

    if vertical_movie_on_horizontal_screen:
        # the code's a bit ugly, reflecting a history when the drawing area had the 9:16 shape
        # and then was expanded at the expense of being partially covered by the timeline and movie list areas
        DRAWING_AREA_Y_SHARE = 1
        DRAWING_AREA_X_SHARE = (screen.get_height() * (res.IWIDTH/res.IHEIGHT)) / screen.get_width()
        DRAWING_AREA_X_START = 0
        DRAWING_AREA_Y_START = 0
        timeline_rect = (DRAWING_AREA_X_SHARE, 0, 1-DRAWING_AREA_X_SHARE, TIMELINE_Y_SHARE)

        MOVIES_X_START = DRAWING_AREA_X_SHARE
        MOVIES_X_SHARE = 1-TOOLBAR_X_SHARE-DRAWING_AREA_X_SHARE-LAYERS_X_SHARE
        TOOLBAR_X_START = DRAWING_AREA_X_SHARE + MOVIES_X_SHARE

        DRAWING_AREA_X_SHARE = 1 - TOOLBAR_X_SHARE - LAYERS_X_SHARE
    else:
        DRAWING_AREA_X_SHARE = 1 - TOOLBAR_X_SHARE - LAYERS_X_SHARE
        DRAWING_AREA_Y_SHARE = DRAWING_AREA_X_SHARE # preserve screen aspect ratio
        DRAWING_AREA_X_START = TOOLBAR_X_SHARE
        DRAWING_AREA_Y_START = TIMELINE_Y_SHARE
        timeline_rect = (0, 0, 1, TIMELINE_Y_SHARE)
        # this is what MOVIES_Y_SHARE is in horizontal layouts; vertical just inherit it
        #MOVIES_Y_SHARE = 1-DRAWING_AREA_Y_SHARE-TIMELINE_Y_SHARE
        MOVIES_X_START = TOOLBAR_X_SHARE
        MOVIES_X_SHARE = 1-TOOLBAR_X_SHARE-LAYERS_X_SHARE
        TOOLBAR_X_START = 0

    MOVIES_Y_START = 1 - MOVIES_Y_SHARE
    LAYERS_X_START = 1 - LAYERS_X_SHARE
    LAYERS_Y_START = TIMELINE_Y_SHARE

    last_tool = layout.full_tool if layout else None

    layout = Layout()
    layout.add((DRAWING_AREA_X_START, DRAWING_AREA_Y_START, DRAWING_AREA_X_SHARE, DRAWING_AREA_Y_SHARE), DrawingArea(vertical_movie_on_horizontal_screen))

    layout.add(timeline_rect, TimelineArea())
    layout.add((MOVIES_X_START, MOVIES_Y_START, MOVIES_X_SHARE, MOVIES_Y_SHARE), MovieListArea())
    layout.add((LAYERS_X_START, LAYERS_Y_START, LAYERS_X_SHARE, LAYERS_Y_SHARE), LayersArea())

    tools_width_height = [
        ('pen', 0.23, 1),
        ('pencil', 0.23, 1),
        ('eraser-big', 0.23, 0.88),
        ('eraser-medium', 0.18, 0.68),
        ('eraser', 0.13, 0.48),
    ]
    offset = 0
    for tool, width, height in tools_width_height:
        layout.add((TOOLBAR_X_START+offset*0.15,0.85+(0.15*(1-height)),width*0.15, 0.15*height), ToolSelectionButton(TOOLS[tool]))
        offset += width
    color_w = 0.025*2
    i = 0

    layout.add((TOOLBAR_X_START+color_w*2.4, 0.86, color_w*.6, color_w*1.2), ToolSelectionButton(TOOLS['needle']))
    layout.add((TOOLBAR_X_START+color_w*1.9, 0.85-color_w, color_w*1.1, color_w*1.7), ToolSelectionButton(TOOLS['tweezers']))
    
    first_xy = None
    for row,y in enumerate(np.arange(0.25,0.85-0.001,color_w)):
        for col,x in enumerate(np.arange(0,0.15-0.001,color_w)):
            if first_xy is None:
                first_xy = (TOOLBAR_X_START+x,y)
            tool = None
            if row == len(palette.colors):
                if col == 0:
                    tool = TOOLS['zoom']
                elif col == 1:
                    tool = Tool(PaintBucketTool(palette.bg_color), palette.bg_cursor, '')
                else:
                    continue
            if not tool:
                r = len(palette.colors)-row-1
                tool = Tool(PaintBucketTool(palette.colors[r][col], palette.change_color_func(r,col)), palette.cursors[r][col], '')
            if isinstance(tool.tool, PaintBucketTool):
                PaintBucketTool.color2tool[tool.tool.color] = tool
            layout.add((TOOLBAR_X_START+x,y,color_w,color_w), ToolSelectionButton(tool))
            i += 1

    palette_rect = (first_xy[0], first_xy[1], color_w*palette.columns, color_w*palette.rows)
    layout.add(palette_rect, PaletteElem())

    funcs_width = [
        ('insert-frame', 0.33),
        ('remove-frame', 0.33),
        ('play', 0.33)
    ]
    offset = 0
    for func, width in funcs_width:
        if func == 'play':
            button = TogglePlaybackButton(load_image('play.png'), load_image('pause.png'))
        else:
            button = ToolSelectionButton(TOOLS[func])
        layout.add((TOOLBAR_X_START+offset*0.15,0.15,width*0.15, 0.1), button)
        offset += width

    layout.freeze()

    set_tool(last_tool if last_tool else TOOLS['pen'])

def new_movie_clip_dir(): return os.path.join(WD, format_now())

def default_clip_dir():
    clip_dirs = get_clip_dirs(sort_by='st_mtime')
    if not clip_dirs:
        # first clip - create a new directory
        return new_movie_clip_dir(), True
    else:
        return os.path.join(WD, clip_dirs[0]), False

def load_clips_dir():
    movie_dir, is_new_dir = default_clip_dir()
    global movie
    movie = Movie(movie_dir) if is_new_dir else open_movie_with_progress_bar(movie_dir)

    init_layout()
    movie.restore_viewing_params()

    global movie_list
    movie_list = MovieList()

class SwapWidthHeightHistoryItem(HistoryItemBase):
    def __init__(self): HistoryItemBase.__init__(self, restore_pos_before_undo=False)
    def undo(self):
        swap_width_height(from_history=True)
        return SwapWidthHeightHistoryItem()

def swap_width_height(from_history=False):
    trace.event('swap-width-height')
    res.set_resolution(res.IHEIGHT, res.IWIDTH)
    init_layout()
    movie.fit_to_resolution()
    if not from_history:
        history.append_item(SwapWidthHeightHistoryItem())

def lock_all_layers():
    items = []
    for layer in movie.layers:
        if not layer.locked:
            layer.toggle_locked()
            items.append(ToggleHistoryItem(layer.toggle_locked))
    history.append_item(HistoryItemSet(items))

# The history is "global" for all operations within a movie. In some (rare) animation programs
# there's a history per frame. One problem with this is how to undo timeline
# operations like frame deletions or holds (do you have a separate undo function for this?)
# It's also somewhat less intuitive in that you might have long forgotten
# what you've done on some frame when you visit it and press undo one time
# too many
#

class History:
    # a history is kept per movie. the size of the history is global - we don't
    # want to exceed a certain memory threshold for the history
    byte_size = 0
    
    def __init__(self):
        self.undo = []
        self.redo = []
        layout.drawing_area().clear_fading_mask()
        self.suggestions = None

    def __del__(self):
        for op in self.undo + self.redo:
            History.byte_size -= op.byte_size()

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
        if item is None or item.nop():
            return

        self._merge_prev_suggestions()

        self.undo.append(item)
        History.byte_size += item.byte_size() - sum([op.byte_size() for op in self.redo])
        self.redo = [] # forget the redo stack
        while self.undo and History.byte_size > MAX_HISTORY_BYTE_SIZE:
            History.byte_size -= self.undo[0].byte_size()
            del self.undo[0]

        layout.drawing_area().clear_fading_mask() # new operations invalidate old skeletons

    def undo_item(self, drawing_changes_only):
        trace.event('undo')
        if self.suggestions:
            s = self.suggestions
            self.suggestions = None
            for item in s:
                self.append_item(item)

        if self.undo:
            last_op = self.undo[-1]
            if drawing_changes_only and (not last_op.is_drawing_change() or not last_op.from_curr_pos()):
                return

            if last_op.make_undone_changes_visible():
                return # we had to seek to the location of the changes about to be undone - let the user
                # see the state before the undoing, next time the user asks for undo we'll actually undo
                # and it will be clear what the undoing did

            redo = last_op.undo()
            History.byte_size += redo.byte_size() - last_op.byte_size()
            if redo is not None:
                self.redo.append(redo)
            self.undo.pop()

        layout.drawing_area().clear_fading_mask() # changing canvas state invalidates old skeletons

    def last_item(self):
        return self.undo[-1] if self.undo else None

    def redo_item(self):
        trace.event('redo')
        if self.redo:
            last_op = self.redo[-1]
            if last_op.make_undone_changes_visible():
                return
            undo = last_op.undo()
            History.byte_size += undo.byte_size() - last_op.byte_size()
            if undo is not None:
                self.undo.append(undo)
            self.redo.pop()

        layout.drawing_area().clear_fading_mask() # changing canvas state invalidates old skeletons

    def clear(self):
        History.byte_size -= sum([op.byte_size() for op in self.undo+self.redo])
        self.undo = []
        self.redo = []
        self.suggestions = None

escape = False

user_event_offset = 0
def user_event():
    global user_event_offset
    user_event_offset += 1
    return user_event_offset

MOUSEBUTTONDOWN = user_event()
MOUSEMOTION = user_event()
MOUSEBUTTONUP = user_event()

PLAYBACK_TIMER_EVENT = user_event()
SAVING_TIMER_EVENT = user_event() 
FADING_TIMER_EVENT = user_event()
HISTORY_TIMER_EVENT = user_event()

timer_events = [
    PLAYBACK_TIMER_EVENT,
    SAVING_TIMER_EVENT,
    FADING_TIMER_EVENT,
    HISTORY_TIMER_EVENT,
]

keyboard_shortcuts_enabled = False # enabled by Ctrl-A; disabled by default to avoid "surprises"
# upon random banging on the keyboard

def set_clipboard_image(surface):
  QApplication.clipboard().setPixmap(QPixmap.fromImage(surface.qimage()))

cut_frame_content = None

def copy_frame():
    trace.event('copy-paste')
    global cut_frame_content
    cut_frame_content = movie.curr_frame().get_content()
    set_clipboard_image(movie.curr_frame().surface())

def cut_frame():
    trace.event('copy-paste')
    history_item = HistoryItemSet([HistoryItem('color'), HistoryItem('lines')])

    global cut_frame_content
    frame = movie.edit_curr_frame()
    cut_frame_content = frame.get_content()
    set_clipboard_image(frame.surface())
    frame.clear()

    history_item.optimize()
    history.append_item(history_item)

def paste_frame():
    trace.event('copy-paste')
    if not cut_frame_content:
        return

    history_item = HistoryItemSet([HistoryItem('color'), HistoryItem('lines')])

    movie.edit_curr_frame().set_content(cut_frame_content)

    history_item.optimize()
    history.append_item(history_item)

def open_explorer(path):
    if on_windows:
        # note: attempts to persuade explorer to open in "large icons mode" involve using a COM interface
        # with OS-dependent numeric viewing modes, and even upon success (which is unlikely since you
        # need to look for the window based on the path and sleep until you find it but there may be
        # other windows with the same path and how do you know if you found the new one), it has the
        # downside of scrolling the window up, so the user doesn't see the selection that at least
        # is visible if we just run explorer /select,path. so giving up on forcing it to show icons
        #
        # another advantage of giving up is that Windows remembers the user's preference from the last
        # time the directory was opened and since we always export into the same directory,
        # you only need to configure this once.
        subprocess.Popen(['explorer', '/select,'+path])
    else:
        subprocess.Popen(['nautilus', '-s', path])

def export_and_open_explorer():
    trace.event('export-clip')
    movie.save_and_export()

    open_explorer(movie.gif_path() if len(movie.frames)>1 else movie.still_png_path())

def open_dir_path_dialog():
    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.Directory)
    dialog.setViewMode(QFileDialog.Detail)
    dialog.setAcceptMode(QFileDialog.AcceptOpen)
    dialog.setOptions(QFileDialog.DontUseNativeDialog)
    dialog.setLabelText(QFileDialog.LookIn, '')
    dialog.setLabelText(QFileDialog.FileType, '')
    dialog.setWindowTitle('Open a folder with zero or more Tinymation clips')

    # setEnabled(False) keeps mouse events from getting to the main window (with modality/
    # other we keep getting mouse events though not keyboard events, setEnabled is the only thing
    # that seems to block it; no parent widget set for the dialog or setEnabled(False) disables the dialog, too
    widget.setEnabled(False)
    try:
        if dialog.exec():
            fileNames = dialog.selectedFiles()
            return fileNames[0]
    finally:
        widget.setEnabled(True)

def open_clip_dir():
    trace.event('open-clip-dir')
    file_path = open_dir_path_dialog()
    global WD
    if file_path and os.path.realpath(file_path) != os.path.realpath(WD):
        movie.save_before_closing()
        set_wd(file_path)
        load_clips_dir()

def patching_tool_selected():
    for cls in NeedleTool, PaintBucketTool, PenTool, TweezersTool:
        if isinstance(layout.tool, cls):
            return True

needle_cursor_selected = False

def process_keyup_event(event):
    ctrl = event.modifiers() & Qt.ControlModifier
    global needle_cursor_selected
    if not ctrl and needle_cursor_selected:
        try_set_cursor(layout.full_tool.cursor[0])
        needle_cursor_selected = False

def dump_and_clear_profiling_data():
    print('saving trace data to event-trace/')
    trace.save('event-trace')
    trace.clear()

import repl
def run_repl():
    def on_close():
        widget.repl = None
    window = repl.REPLDialog(parent=widget, namespace=globals(), on_close=on_close)
    window.show()
    widget.repl = window # keep a reference

    # note that this only helps "partially" - we lose the ability to set cursors
    # after _closing_ the REPL window for some reason. it works as long as it's open...
    global needle_cursor_selected
    try_set_cursor(layout.full_tool.cursor[0])
    needle_cursor_selected = False

# TODO: proper test for modifiers; less use of Ctrl
def process_keydown_event(event):
    if event.key() in [ord('/'),ord('?'),Qt.Key_F1]:
        help_screen()
        return

    ctrl = event.modifiers() & Qt.ControlModifier
    shift = event.modifiers() & Qt.ShiftModifier

    global needle_cursor_selected
    if ctrl and not needle_cursor_selected and patching_tool_selected():
        try_set_cursor(needle_cursor[0])
        needle_cursor_selected = True

    # Like Escape, Undo/Redo and Delete History are always available thru the keyboard [and have no other way to access them]
    if event.key() == Qt.Key_Space:
        if ctrl:
            history.redo_item()
            return
        else:
            history.undo_item(drawing_changes_only=True)
            return

    # Ctrl-Z: undo any change (space only undoes drawing changes and does nothing if the latest change in the history
    # isn't a drawing change)
    if event.key() == Qt.Key_Z and ctrl:
        history.undo_item(drawing_changes_only=False)
        return

    # Ctrl-E: export
    if ctrl and event.key() == Qt.Key_E:
        export_and_open_explorer()
        return

    # Ctrl-O: open a directory
    if ctrl and event.key() == Qt.Key_O:
        open_clip_dir()
        return

    # Ctrl-C/X/V
    if ctrl:
        if event.key() == Qt.Key_C:
            copy_frame()
            return
        elif event.key() == Qt.Key_X:
            cut_frame()
            return
        elif event.key() == Qt.Key_V:
            paste_frame()
            return

    # Ctrl-[/Ctrl-]: layer up/down, alias to arrow up/down
    if ctrl and event.key() == Qt.Key_BracketLeft:
        prev_layer()
        return
    if ctrl and event.key() == Qt.Key_BracketRight:
        next_layer()
        return

    # Ctrl-1: alias to 1
    if ctrl and event.key() == Qt.Key_1:
        zoom_to_film_res()
        return

    # Ctrl-R: rotate
    if ctrl and event.key() == Qt.Key_R:
        swap_width_height()
        return

    if ctrl and event.key() == Qt.Key_L:
        lock_all_layers()
        return

    # Ctrl-P: dump profiling data
    if ctrl and event.key() == Qt.Key_P:
        dump_and_clear_profiling_data()
        return

    # Ctrl-D: debug with a REPL
    if ctrl and event.key() == Qt.Key_D:
        run_repl()
        return

    # Ctrl-N: rename
    if ctrl and event.key() == Qt.Key_N:
        rename_clip()
        return

    # Ctrl-2/3/4: set layout to drawing/layers/animation
    if ctrl and event.key() == Qt.Key_2:
        layout.mode = DRAWING_LAYOUT
        return
    if ctrl and event.key() == Qt.Key_3:
        layout.mode = LAYERS_LAYOUT
        return
    if ctrl and event.key() == Qt.Key_4:
        layout.mode = ANIMATION_LAYOUT
        return

    # other keyboard shortcuts are enabled/disabled by Ctrl-A
    global keyboard_shortcuts_enabled

    def quiet_ord(x):
        try:
            return ord(x)
        except:
            return x
    if keyboard_shortcuts_enabled:
        for tool in TOOLS.values():
            if event.key() in list(tool.chars) or event.key() in [quiet_ord(c) for c in tool.chars]:
                set_tool(tool)
                return

        for func, chars in FUNCTIONS.values():
            if event.key() in list(chars) or event.key() in [quiet_ord(c) for c in chars]:
                func()
                return
                
    if event.key() == Qt.Key_A and ctrl:
        keyboard_shortcuts_enabled = not keyboard_shortcuts_enabled
        print('Ctrl-A pressed -','enabling' if keyboard_shortcuts_enabled else 'disabling','keyboard shortcuts')

#layout.draw()
#pg.display.flip()

UNDER_TEST = os.getenv('TINYTEST') is not None

if UNDER_TEST:
    import tinytest
    from PySide6.QtCore import QThread, Signal, QObject, Slot

    # Worker object to read socket in a thread
    class SocketReader(QObject):
        # Signal to send arbitrary Python object
        data_received = Signal(object)
    
        def __init__(self, conn):
            super().__init__()
            self.conn = conn
    
        def run(self):
            """Run in worker thread to read socket."""
            while True:
                try:
                    # Read from socket (any picklable object)
                    message = self.conn.recv()
                    if message == 'shutdown':
                        break
                    # Emit signal with the object
                    self.data_received.emit(message)
                except (BrokenPipeError, EOFError):
                    break
                except Exception as e:
                    break

    class TestEvent:
        def accept(self):
            self.conn.send(('done', self.command))

    class TestTabletEvent(TestEvent):
        def __init__(self, command):
            event, x, y, self.p, self.time = command
            self.t = {'tablet-press':QEvent.TabletPress, 'tablet-move':QEvent.TabletMove, 'tablet-release': QEvent.TabletRelease}[event]
            self.xy = QPoint(x,y)
        def pressure(self): return self.p
        def type(self): return self.t
        def position(self): return self.xy
        def timestamp(self): return self.time

    def handle_test_event(widget, command):
        event = TestTabletEvent(command)
        event.command = command
        event.conn = widget.test_conn

        widget.tabletEvent(event)

# Define the Windows MSG structure for ctypes
class MSG(ct.Structure):
    _fields_ = [
        ('hwnd', ct.wintypes.HWND),
        ('message', ct.wintypes.UINT),
        ('wParam', ct.wintypes.WPARAM),
        ('lParam', ct.wintypes.LPARAM),
        ('time', ct.wintypes.DWORD),
        ('pt', ct.wintypes.POINT),
    ]

# Windows message and flag constants
WM_TABLET_QUERYSYSTEMGESTURESTATUS = 0x02CC

TABLET_DISABLE_PRESSANDHOLD = 0x00000001  # Core flag to disable press-and-hold
TABLET_DISABLE_PENTAPFEEDBACK = 0x00000008
TABLET_DISABLE_PENBARRELFEEDBACK = 0x00000010
TABLET_DISABLE_FLICKS = 0x00010000

class TinymationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Tinymation')
        
        # Get the size of the primary screen
        scr = QGuiApplication.primaryScreen().geometry()
        self.setGeometry(scr)
        
        # Create backing store QImage with the scr size
        self.image = screen.qimage_unsafe()
        self.showFullScreen()
        
        # Enable tablet tracking
        #self.setAttribute(Qt.WidgetAttribute.WA_TabletTracking)

        self.timers = []
        # we save the current frame every 15 seconds
        for event, rate in ((PLAYBACK_TIMER_EVENT, 1000/FRAME_RATE), (SAVING_TIMER_EVENT, 15*1000), (FADING_TIMER_EVENT, 1000/FADING_RATE)):
            timer = QTimer(self)
            timer.timeout.connect(lambda event=event: self.on_timer(event))
            timer.start(rate)
            self.timers.append(timer)

    def start_loading(self):
        load_clips_dir()
        global history
        history = History()
        layout.draw()
        self.redrawScreen()

        if UNDER_TEST:
            self.start_test_slave_thread()

    def start_test_slave_thread(self):
        # Setup socket reader thread
        self.test_conn = tinytest.start_testing()
        self.test_thread = QThread()
        self.test_reader = SocketReader(self.test_conn)
        self.test_reader.moveToThread(self.test_thread)
        # Connect signals
        self.test_reader.data_received.connect(self.test_command)
        self.test_thread.started.connect(self.test_reader.run)
        # Start thread
        self.test_thread.start()

    def on_timer(self, event):
        if layout is None:
            return
        class Event: pass
        e = Event()
        e.type = event
        with trace.start({PLAYBACK_TIMER_EVENT:'playback-timer', SAVING_TIMER_EVENT:'saving-timer', FADING_TIMER_EVENT: 'fading-timer'}[event]):
            layout.on_event(e)
            if self.redrawLayoutIfNeeded(e):
                self.redrawScreen()

    def redrawScreen(self):
        if layout and layout.update_roi is not None:
            if layout.update_roi[2] > 0 and layout.update_roi[3] > 0:
                self.repaint(*layout.update_roi)
            layout.update_roi = None
        else:
            self.update()

    def redrawLayoutIfNeeded(self, event=None):
        if event is None or (layout.is_playing and event.type == PLAYBACK_TIMER_EVENT) or layout.drawing_area().redraw_fading_mask or event.type not in timer_events and not movie_list.opening:
            layout.draw()
            layout.drawing_area().redraw_fading_mask = False
            if not layout.is_playing:
               cache.collect_garbage()
            return True

    def paintEvent(self, event):
        with trace.start('paint'):
            painter = QPainter(self)
            rect = event.rect()
            painter.drawImage(rect, self.image, rect)
            painter.end()
            event.accept()

    def mouseEvent(self, event, type):
        with trace.start({MOUSEBUTTONDOWN:'mouse-down',MOUSEMOTION:'mouse-move',MOUSEBUTTONUP:'mouse-up'}[type]):
            class Event:
                pass
            e = Event()
            e.type = type
            pos = event.position()
            e.pos = (pos.x(), pos.y())
            e.subpixel = False
            e.time = event.timestamp()
            layout.on_event(e)
            if self.redrawLayoutIfNeeded(event):
                self.redrawScreen()
            event.accept()

    def mousePressEvent(self, event): self.mouseEvent(event, MOUSEBUTTONDOWN)
    def mouseMoveEvent(self, event): self.mouseEvent(event, MOUSEMOTION)
    def mouseReleaseEvent(self, event): self.mouseEvent(event, MOUSEBUTTONUP)

    def tabletEvent(self, event):
        class Event:
            pass
        e = Event()
        e.type = {QEvent.TabletPress: MOUSEBUTTONDOWN, QEvent.TabletMove: MOUSEMOTION, QEvent.TabletRelease: MOUSEBUTTONUP}.get(event.type())
        with trace.start({MOUSEBUTTONDOWN:'stylus-down',MOUSEMOTION:'stylus-move',MOUSEBUTTONUP:'stylus-up'}[e.type]):
            pos = event.position()
            # there seems to be a disagreement in "where the center of the pixel is" - at integer coordinates 0,1,2...
            # or at 0.5, 1.5, 2.5... between the tablet events and the coordinate system of xy2frame/frame2xy. I didn't give
            # it much thought after observing that the below seems to work in the sense of drawing reasonably "exactly"
            # where the cursor hotspot is (which isn't happening without this correction)
            e.pos = (pos.x()-.5, pos.y()-.5)
            e.subpixel = True
            e.pressure = event.pressure()
            e.time = event.timestamp()
            layout.on_event(e)
            if self.redrawLayoutIfNeeded(event):
                self.redrawScreen()
            event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.shutdown()
            self.close()
            return

        if layout.is_pressed:
            return # ignore keystrokes (except ESC) when a mouse tool is being used

        if layout.is_playing and not (keyboard_shortcuts_enabled and event.key() in [Qt.Key_Enter, Qt.Key_Return]):
            return # ignore keystrokes (except ESC and ENTER) during playback

        with trace.start('key-down'):
            process_keydown_event(event)
            if self.redrawLayoutIfNeeded(event):
                self.redrawScreen()
            event.accept()

    def keyReleaseEvent(self, event):
        with trace.start('key-up'):
            process_keyup_event(event)

    # on Windows, this is necessary to disable touch "gestures" such as expanding and then disappearing circles
    # adding a delay to the input
    def nativeEvent(self, eventType, message):
        if eventType == b"windows_generic_MSG":
            # Cast the message pointer to MSG structure
            msg = MSG.from_address(int(message))
            if msg.message == WM_TABLET_QUERYSYSTEMGESTURESTATUS:
                # Return the disable flag(s); you can bitwise-OR multiple if needed
                return True, TABLET_DISABLE_PRESSANDHOLD | TABLET_DISABLE_PENTAPFEEDBACK | TABLET_DISABLE_PENBARRELFEEDBACK
        # Fall back to default handling for other events
        return super().nativeEvent(eventType, message)

    def shutdown(self):
        if UNDER_TEST:
            self.test_thread.quit()
            self.test_thread.wait()
            self.test_conn.close()

        movie.save_before_closing()

    if UNDER_TEST:
        @Slot(object)
        def test_command(self, command):
            handle_test_event(self, command)

widget = TinymationWidget()
try_set_cursor(pen_cursor[0])

def signal_handler(sig, frame):
    print("\nInterrupted by Ctrl+C! Shutting down, please wait...")
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    widget.shutdown()
    QApplication.quit()
signal.signal(signal.SIGINT, signal_handler)

QTimer.singleShot(0, widget.start_loading)
status = app.exec()
dump_and_clear_profiling_data()

delete_lock_file()

surf.show_stats()

sys.exit(status)

