#!/bin/bash
set -ex
g++ -shared -o tinylib.so -O3 -g tinylib.cpp floodfill.cpp -Wall -Werror -fPIC
mv tinylib.so ../pygame
