import sys, os
import subprocess

def cmd(c):
    print(f'> {c}')
    status = os.system(c)
    if status:
        raise Exception(f'`{c}` failed with status {status}')

cmd('pyinstaller --exclude viztracer --name Tinymation --windowed --icon assets/icon.ico tinymation.py')
cmd('cp -a ../gifski/gifski-win.exe dist/Tinymation/_internal')
cmd('cp -a assets dist/Tinymation')
cmd('cp -a tinylib.dll dist/Tinymation')

# this part is for TBB DLLs which pyinstaller misses
try:
    os.makedirs('dist/Tinymation/_internal/Library/bin')
except:
    pass
cmd(f'cp -a {sys.prefix}/Library/bin/tbb*.dll dist/Tinymation/_internal/Library/bin')

print()
print('removing unnecessary files')
cmd(f'rm -f dist/Tinymation/_internal/cv2/opencv_videoio_ffmpeg4100_64.dll')

QtDeps = '''
QtCore.pyd
QtWidgets.pyd
QtGui.pyd
Qt6Widgets.dll
Qt6Gui.dll
VCRUNTIME140_1.dll
VCRUNTIME140.dll
MSVCP140.dll
pyside6.abi3.dll
MSVCP140_2.dll
MSVCP140_1.dll
Qt6Core.dll
plugins/generic/qtuiotouchplugin.dll
plugins/platforminputcontexts
plugins/platforminputcontexts/qtvirtualkeyboardplugin.dll
plugins/platforms/qminimal.dll
plugins/platforms/qwindows.dll
plugins/platforms/qdirect2d.dll
plugins/platforms/qoffscreen.dll
plugins/imageformats/qico.dll
'''.strip().split()

qtroot = 'dist/Tinymation/_internal/PySide6/'
files = subprocess.getoutput(f'find {qtroot} -type f').strip().split()
for f in files:
    name = f[len(qtroot):]
    if name not in QtDeps:
        cmd(f'rm -f {f}')


