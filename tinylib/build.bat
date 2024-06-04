#cl vectorize.cpp /arch:AVX /O2 /FAss /Qvec-report:2  /openmp:experimental
#cl *.cpp /EHsc /I "C:\Program Files (x86)\Intel\oneAPI\tbb\2021.12\include" /openmp:experimental /arch:AVX /O2 /LD /Fetinylib.dll /link /DEF:tinylib.def /LIBPATH:"C:\Program Files (x86)\Intel\oneAPI\tbb\2021.12\lib"
cl *.cpp /EHsc /O2 /LD /Fetinylib.dll /link /DEF:tinylib.def
move tinylib.dll ../pygame/tinylib.dll
