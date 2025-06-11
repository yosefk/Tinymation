
IWIDTH = 1920
IHEIGHT = 1080

def set_resolution(w, h):
    global IWIDTH
    global IHEIGHT
    IWIDTH, IHEIGHT = w, h

def clip(x, y):
    return min(IWIDTH,max(x,0)), min(IHEIGHT,max(y,0))
