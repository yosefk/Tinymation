import sys, os

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
