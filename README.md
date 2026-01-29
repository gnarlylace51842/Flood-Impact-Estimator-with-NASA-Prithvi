# Flood-Impact-Estimator-with-NASA-Prithvi
A project I made testing my skills on Image Segmentation with TF2 (Machine Learning Principle) (A REAL SEGMENTATION PIPELINE!)
It essentially takes a satellite picture, performs image segmentation on it, and returns a binary mask which we can use for post-processing later (if I ever implement it 😭)

To use:
1. Provide a .tif image file to use.
2. Put everything under one folder, the privthi_mask, the .tif file, the prithvi .venv, and where you want your outputs.
3. Use terminal to run the program with the desired .tif file (required so from argParse)
4. After, you should get two images in your outputs:
   a. an image.png, which is a normal RGB visualization of your .tif file
   b. and mask.png, a binary segmentation mask (basically black and white to show differences)

White is where the program detected foreground, and everywhere else is black (where it could be the wrong class index/band order/image type)

Might use this later if I need it for specific downstream steps later in the line :)
