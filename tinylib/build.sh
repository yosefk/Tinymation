#!/bin/bash
set -ex
g++ -std=c++17 -shared -o tinylib.so -O3 -g *.cpp fitpack/*.cpp -Wall -Werror -fPIC -Wl,--version-script=tinylib.expmap
mv tinylib.so ../pyside
