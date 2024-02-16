#!/bin/bash
set -ex
gcc -shared -o tinylib.so -O3 -g tinylib.c -Wall -Werror -fPIC
mv tinylib.so ../pygame
