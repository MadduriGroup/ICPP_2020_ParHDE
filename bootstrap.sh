#!/bin/sh

wget https://github.com/sbeamer/gapbs/archive/master.zip
unzip master.zip -d gapbs
rsync --exclude=util.h --exclude=platform_atomics.h gapbs/gapbs-master/src/*.h gapbs/
