#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/flickr_logo/models/solver.prototxt
