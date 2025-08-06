#!/bin/bash
set -ex
# to build *.ispc files:
#~/ispc/ispc-v1.27.0-linux/bin/ispc -O3 --addressing=64 --PIC --emit-asm --target=sse2-i32x4,sse4-i32x8,avx2-i32x8,avx512skx-x16 blit.ispc -o blit.lin.s
TBB=/opt/intel/oneapi/tbb/latest
cp $TBB/lib/libtbb.so .
g++ -std=c++17 -shared -o tinylib.so -O3 -g -I$TBB/include *.cpp *.lin*.s fitpack/*.cpp -Wall -Werror -fPIC -Wl,--version-script=tinylib.expmap -fno-exceptions -fno-rtti libtbb.so
mv tinylib.so libtbb.so ../pyside
