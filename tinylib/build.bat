
rem cl vectorize.cpp /arch:AVX /O2 /FAss /Qvec-report:2  /openmp:experimental
rem cl *.cpp /EHsc /I "C:\Program Files (x86)\Intel\oneAPI\tbb\2021.12\include" /openmp:experimental /arch:AVX /O2 /LD /Fetinylib.dll /link /DEF:tinylib.def /LIBPATH:"C:\Program Files (x86)\Intel\oneAPI\tbb\2021.12\lib"
rem ispc.exe -O3 --addressing=64  --target=sse2-i32x4,sse4-i32x8,avx2-i32x8,avx512knl-x16,avx512skx-x16 blit.ispc -o blit.win.obj
rem C:\Users\Yossi\Downloads\ispc-v1.27.0-windows\ispc-v1.27.0-windows\bin\ispc.exe -O3 --addressing=64  --target=sse2-i32x4,sse4-i32x8,avx2-i32x8,avx512skx-x16 blit.ispc -o blit.win.obj
rem f2c -A -a -f -r8 
set TBB="C:\Program Files (x86)\Intel\oneAPI\tbb\2021.12"
cl /std:c++17 /I. /I./f2c-include /I%TBB%/include *.cpp *.win*.obj fitpack/*.cpp /EHsc /O2 /LD /Fetinylib.dll /link /DEF:tinylib.def /LIBPATH:%TBB%\lib
move tinylib.dll ../pyside/tinylib.dll
