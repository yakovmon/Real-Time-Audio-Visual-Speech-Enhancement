'''
An example using Scipy library.
'''
from scipy.misc import imread, imresize, imshow

# Read an jpg image into a numpy array.
img = imread('cat.jpg')

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3).
# we multiply it by the array [1, 0.95, 0.9] of shape (3,).
img_tinted = img * [0, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))

# Show the original image.
imshow(img)

# Show the tinted image.
imshow(img_tinted)
