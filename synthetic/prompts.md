# Pompts for Synthetic COCO Dataset Generation

## Prompt 1: Generate Synthetic COCO Dataset for Object Detection

I have a set of shapes that i will laser cut on mdf board, then paint white. These objects are roughly 10 cm and I want to detect one or more objects held by one or more people in front of a webcam. I want to create a synthetic COCO dataset to train an object recognition model (mediapipe object detector).

The classes are:
    0: background
    1: baton
    2: clubs
    3: coin
    4: diamond
    5: heart

Write a python script to generate synthetic images and corresponding COCO annotations for these classes. The script should create images with random placements, rotations, and scales of the objects against various backgrounds. Ensure that the annotations are in the correct COCO format, including bounding boxes and class labels.
Use procedural backgrounds that approximate visual complexity — variation in color, texture, lighting, gradients, or noise.
Use the shape images from the following paths as the object templates:

    - baton: assets/shapes/baton.png
    - clubs: assets/shapes/clubs.png
    - coin: assets/shapes/coin.png
    - diamond: assets/shapes/diamond.png
    - heart: assets/shapes/heart.png

