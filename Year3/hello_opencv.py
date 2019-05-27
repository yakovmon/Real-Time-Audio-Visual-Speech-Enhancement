'''
An example using opencv library.
'''

import cv2

# The function imread loads an image from the specified file and returns it.
image = cv2.imread('cat.jpg')

# The picture window name.
window_name = "My_Picture"

#  The function namedWindow creates a window that can be used as a placeholder for images.
#  Created windows are referred to by their names.
cv2.namedWindow(window_name)

# The function imshow displays an image in the specified window.
cv2.imshow(window_name, image)

# The function waitKey waits for a key event infinitely or for delay.
cv2.waitKey(0)

# The function destroyAllWindows destroys all of the opened HighGUI windows.
cv2.destroyAllWindows()
