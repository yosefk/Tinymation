icons = [
    ('sheets', 0, 40),
    ('garbage', 0, 40),
    ('pencil-tool', 90, 20),
    ('pen-tool', 90, 20),
    ('tweezers-tool', 0, 50),
    ('needle-tool', 0, 50),
    ('zoom-tool', 0, 40),
    ('locked', 0, 30),
    ('unlocked', 0, 30),
    ('play', 0, 40),
    ('pause', 0, 40),
    ('eraser-tool', -90, 32),
    ('water-tool', 0, 30),
    ('paint_bucket', 0, 60),
    ('splash-1', 0, 40),
    ('splash-14', 0, 40),
    ('splash-11', 0, 40),
    ('hold_yellow', 0, 25),
    ('no_hold', 0, 45),
    ]
help_html = '''
<html>
<head>
    <style>
        body { font-size: 18px; }
        tt { font-size: 22px; font-weight: bold }
    </style>
</head>
<body>

To create a new movie, frame, or layer, click the <b>New</b> button %(sheets)s and then click the selected movie, frame or layer (marked by a thick border & big round corners.) Click elsewhere to get the previous tool.</li>
You can similarly <b>Delete</b> the selected movie, frame or layer %(garbage)s. Each movie lives in its own folder (Ctrl-E shows these folders.) To un-delete movies, remove <tt>-deleted</tt> from folder names & restart Tinymation. But frames & layers deleted individually are <i>NOT</i> kept around - you can only undo the deletions while they're in the history.</li>

<h1>Keyboard shortcuts</h1>

<font color="green"><b>Green</b></font> shortucts work for very young beginners. The rest is for teachers or experienced users.

<ul>
<li><font color="green"><tt>Space</tt></font> - <b>Undo</b> changes <i>to the current drawing</i></li>
<ul>
   <li><tt>Ctrl-Z</tt> - Undo <i>any</i> changes in any layers / frames. Space is easier to press, and does less damage upon "rage-pressing"</li>
   <li><font color="green"><tt>Ctrl-Space</tt></font> - <b>Redo</b>. Warn beginners to redo <b>right away</b> - like in most programs, <i>doing</i> after <i>undoing</i> makes <i>redoing</i> impossible!</li>
</ul>
<li><tt><font color="purple">Ctrl-A</font></tt> - enables "<b><font color="purple">Advanced</font></b>" shortcuts disabled by default (too easy to press by accident!)</li>
<ul>
<li>Drawing:</li>
<ul>
<li><tt>S</tt> - <b>Soft pencil</b> %(pencil_tool)s for rough drawings</li>
<li><tt>B</tt> - <b>Brush</b> %(pen_tool)s a pen for crisp lines / cleaning up roughs in an upper layer</li>
<li><tt>M</tt> - <b><font color="purple">Modify the last pen line</font></b> with the "tweezers" %(tweezers_tool)s - put the line exactly where you want it. <i><b>Only works right after the line is drawn</b></i> (or with everything done afterwards undone!)</li>
<li><tt>N</tt> - <b><font color="purple">Needle %(needle_tool)s for "patching holes" in pen lines</font></b>, so paint bucket color won't "leak out" of the shape. To patch:</li>
<ul>
  <li><tt>Click</tt> near a line to see little "streams" going thru holes in that line (including basically invisible ones)</li>
  <li><tt>Ctrl-Click</tt> near a stream - a line will appear connecting the 2 "banks" between which it flows (tweezers can edit this line)</li>
  <li><font color="green"><tt>Ctrl</tt></font> (enabled by default) - switches to needle, releasing Ctrl returns the previous tool. A click with Ctrl connects the banks of the closest stream, or shows streams if none appear</li>
</ul>
<li><tt>E</tt> - <b>Eraser</b> %(eraser_tool)s to erase pencil & pen lines - but not color, see D</li>
<ul>
  <li><tt>W</tt> - a smaller eraser, a key to the left of E</li> 
  <li><tt>R</tt> - a bigger eraser, a key to the right of E</li>
</ul>
<li><tt>D</tt> - selects the water bucket like the water <b>drops</b> button %(water_tool)s to <b>delete</b> color in closed shapes</li>
<li><tt>C</tt> or <tt>K</tt> - the paint bu<b>ck</b>et %(paint_bucket)s with the last used <b>color</b></li>
<ul>
<li><tt>Shift-Click</tt> on a color splash icon %(splash_1)s %(splash_14)s %(splash_11)s to change its color (each movie keeps its own palette)</li>
</ul>
<li><tt>L</tt> - <b>Lock %(locked)s</b> / unlock %(unlocked)s the current layer. Locking shows "the real <b>look</b>" of <i>all</i> layers (lower & upper layers no longer tinted in blue/orange)</li>
</ul>
<li><tt>Z</tt> - <b>Zoom</b> %(zoom_tool)s around the clicked point (drag up to zoom in/down to zoom out)</li>
<ul>
<li><font color="green"><tt>Alt</tt></font> (enabled by default) - zoom, release Alt to get the previous tool</li>
<li><tt>1</tt> or <tt>Ctrl-1</tt> - <b>1x zoom</b> around the stylus position</li>
</ul>
<li>Timeline:</li>
<ul>
<li><tt>Enter</tt> - <b>Play %(play)s / pause %(pause)s</b></li>
<li><tt>►</tt> or <tt>&gt;</tt> - Go to the <b>next frame</b></li>
<li><tt>◄</tt> or <tt>&lt;</tt> - Go to the <b>previous frame</b> </li>
<li><tt>▲</tt> - Go one <b>layer up</b></li>
<li><tt>▼</tt> - Go one <b>layer down</b></li>
<li><tt>H</tt> - Toggle "frame <b>hold</b>" (in the timeline window, this "<b>hooks</b>" %(hold_yellow)s or unhooks %(no_hold)s the previous frame, so this one does/doesn't repeat it.) Hold <b>hides</b> rather than deletes the current frame - toggle it back to see the hidden drawing again (handy to turn an inbetween on / off to see what it does, for example.) However, deleting a held frame <i>will</i> delete that hidden drawing, too</li>
<li><tt>+</tt> or <tt>=</tt> - <b>insert</b> (and move to) a new frame %(sheets)s after the current one (which is held in all layers except the current.) This means you can't insert a frame <i>before the first one</i> (you need to insert one <i>after</i> it, and use Ctrl-X/V to move the drawings to the new frames.) But in loop mode, inserting after the last frame is the same as inserting before the first!</li>
<li><tt>-</tt> or <tt>_</tt> - <b>remove</b> a frame %(garbage)s in all layers</li>
</ul>
</ul>
<li>"Advanced" shortcuts enabled by default (no need for Ctrl-A):</li>
<ul>
<li><tt>Ctrl-C/X/V</tt> - <b>Copy/Cut/Paste</b> the drawing (current frame & layer.) Helps move stuff around when students draw into the wrong frames. You can paste <i>into</i> (but not <i>from</i>) other programs - Paint/Krita/...</li>
<li><tt>Ctrl-R</tt> - <b>Rotate</b> the movie (from 1920x1080 to 1080x1920 or back.) Works at any time (though you usually decide up front...)
<li><tt>Ctrl-N</tt> - <b>Name</b> the movie's folder (by default, named by the creation date)
<li><tt>Ctrl-E</tt> - <b>Export</b> the movie (into MP4 & GIF - or PNG for still drawings), and show the exported files in a file viewer. File appear <i>near</i> the movie folder, sharing its name. <i>Inside</i> the movie folder, a transparent PNG sequence appears (frame0000.png, frame0001.png...) which you can import into Krita/Photoshop/...</li>
<li><tt>Ctrl-O</tt> - <b>Open</b> a "main" folder to put movies into. Tinymation shows all the movies in a folder - Ctrl-E shows this folder; Ctrl-O lets you open another one. Useful to open a student's folderyou've copied, or to create a separate folder for each of several people using a computer</li>
<li><tt>Ctrl-2/3/4</tt> - <b>Switch to drawing / layers / animation (default) layout</b>. Ctrl-3 hides the timeline; Ctrl-2 also hides the layers. Useful to avoid distractions in first beginners' lessons</li>
</ul>
</html>
'''
import sys
import io
from PySide6.QtWidgets import QDialog, QTextEdit, QVBoxLayout, QApplication
from PySide6.QtCore import Qt, QBuffer, QByteArray
import base64

def image_to_base64(image):
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QBuffer.WriteOnly)
        image.save(buffer, "PNG")  # Save as PNG to buffer
        buffer.close()

        # Convert to base64
        base64_image = base64.b64encode(byte_array.data()).decode("utf-8")
        return f'<img src="data:image/png;base64,{base64_image}"; style="vertical-align: middle">'

class HelpDialog(QDialog):
    def __init__(self, images, parent=None):
        images = dict([(name, image_to_base64(image)) for name, image in images.items()])
        super().__init__(parent)
        self.setWindowTitle("Tinymation - Help")
        self.setModal(True)
        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen)

        layout = QVBoxLayout(self)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setHtml(help_html % images)

        layout.addWidget(self.text_edit)

