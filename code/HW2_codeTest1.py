from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

# Three dimensional array of ints to store cluster assignements: height x width x cluster 
# Image is 602 x 805 (known in advance) and K means will be run 100 iterations
assignment = np.zeros((603,806,100), dtype=int)

# Three dimensional array of floats to store probabilities that each pixel is a hand
# This is the final probability map output
output = np.zeros((603,806), dtype=float)

# Fuction to resize image 
def resize(imgage, percent):
    scale_percent = percent # percent of original size
    width = int(imgage.shape[1] * scale_percent / 100)
    height = int(imgage.shape[0] * scale_percent / 100)
    dim = (width, height)
    imgResizedTemp = cv2.resize(imgage, dim, interpolation = cv2.INTER_AREA)
    return imgResizedTemp

# Load original hand image
imgOriginal = cv2.imread('hand9.jpg')

# Reduce image by 80% because my cell phone pictures are massive (5+ MB per photo), then convert to RGB and float 32 required for k means
imgResized = resize(imgOriginal, 20)
imgConverted = cv2.cvtColor(imgResized, cv2.COLOR_BGR2RGB)
imgForProcessing = imgConverted.reshape((-1, 3))
imgForProcessing = np.float32(imgForProcessing)

# Display image as a preview, then distroy
cv2.imshow('Image Preview (Reduced Size)', imgResized)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

# Criteria for K means:
# Max iterations = 10, epsilon (accuracy of convergence) = 1.0
# These are the default recommended values for RBG segmentation from OpenCV documentation
# User defines k (I will be using 3, which works best for me)
print('\n--------------------Start of Program--------------------')
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
print('NOTE: For darker skinned individuals, use a higher K value')
k = int(input('Enter desired K (Default = 3. I recommend 3-5): '))
iteration = 0

# Run K means 1 time with the criteria and K from above, and use random initialization centers
_, labels, (centers) = cv2.kmeans(imgForProcessing, k, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)

# Process results for mapping
centers = np.uint8(centers)
labels = labels.flatten()
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(imgConverted.shape)

# Take color of the pixel at (0,0) and call it the background, this is a known based on the image
background = segmented_image[0][0]

# Display centers from first K means iteration
print('\n---Sample data from first K means run (for demonstration purposes)---')
print('Centers are: \n', centers)
print('\nBackground is: ', background)

# Loop through all pixels and record cluster assignment into assignment array
for x in range(0, 603):
    for y in range (0, 806):
        tmp = segmented_image[y][x]
        if(tmp[0] == background[0]):
            assignment[x][y][iteration] = 0
        else:
            assignment[x][y][iteration] = 1


# Some checks for pixels
print('\nPixel (0,0) = ', 'Background' if(assignment[0][0][0] == 0) else 'Skin')
print('Pixel (200,600) = ', 'Background' if(assignment[200][600][0] == 0) else 'Skin')
print('Pixel (500,700) = ', 'Background' if(assignment[500][700][0] == 0) else 'Skin')
print('Pixel (300,700) = ', 'Background' if(assignment[300][700][0] == 0) else 'Skin')
print('-----------------------------------------------------------------------')


# After first K means run, run it again 99 more times with different random initializations
# Store results in assignment array for each iteration
print('\nRunning K means 99 more times with random initializations...')
for z in range(1,100):
    iteration += 1
    _, labels, (centers) = cv2.kmeans(imgForProcessing, k, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(imgConverted.shape)
    background = segmented_image[0][0]
    for x in range(0, 603):
        for y in range (0, 806):
            tmp = segmented_image[y][x]
            if(tmp[0] == background[0]):
                assignment[x][y][iteration] = 0
            else:
                assignment[x][y][iteration] = 1

# Construct probability map based on assignemnts
print('\nCreating probability map with output...')
sumTmp = 0
for x in range(0,603): 
    for y in range (0,806): 
        for z in range (0, 100): 
            if (assignment[x][y][z] == 1):
                sumTmp += 1
            pixelProb = sumTmp / 100
            output[x][y] = pixelProb
        sumTmp = 0


print('\n---Sample data from 100 K means runs (for demonstration purposes)---')
# Should be background every time
print('Probability of pixel (0,0) being skin is: ', output[0][0])
# Should be skin every time
print('Probability of pixel (300,600) being skin is: ', output[300][600])
# Should be contested
print('Probability of pixel (325,774) being skin is: ', output[325][774])
print('----------------------------------------------------------------------')

# Display plotted segmented image
plt.imshow(segmented_image)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

# Allow user to interact with probability map output 
xInput = 0
yInput = 0
print('\nCheck probabilites. Enter X and Y pixel, enter -1 for either to exit')
while (xInput != -1):
    xInput = int(input('Enter x pixel (0-602): '))
    if(xInput != -1):
        yInput = int(input('Enter y pizel (0-806): '))
        if (yInput != -1):
            print('Probability of pixel (', xInput, ', ', yInput, ') being skin is: ', output[xInput][yInput])

