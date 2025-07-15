#!/bin/bash
set -ex
# to build *.ispc files:
# ~/ispc/ispc-v1.27.0-linux/bin/ispc -O3 --addressing=64 --PIC --emit-asm --target=avx2-i32x8 blit.ispc -o blit.s
g++ -std=c++17 -shared -o tinylib.so -O3 -g *.cpp *.lin.s fitpack/*.cpp -Wall -Werror -fPIC -Wl,--version-script=tinylib.expmap
mv tinylib.so ../pyside
