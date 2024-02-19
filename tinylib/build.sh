#!/bin/bash
set -ex
g++ -shared -o tinylib.so -O3 -g tinylib.cpp floodfill.cpp fitpack.cpp splev.cpp fpbspl.cpp -Wall -Werror -fPIC
mv tinylib.so ../pygame
