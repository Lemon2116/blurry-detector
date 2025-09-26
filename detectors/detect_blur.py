from imutils import paths
import argparse
import cv2
import os

# This Python script is used to detect the blurriness of one image, and split them into "blurry", "clear" and "other".
# It collects the classified file names into 3 text files for future categorization.  
# Steps: 
# 1. Open up the terminal
# 2. Type this line into the terminal window, and hit "enter": 
# "python detect_blur.py --images path-to-the-image-folder "

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=260.0,
                help="focus meausers that fall below this value will be considered 'blurry'")
ap.add_argument("-c", "--clear_threshold", type=float, default=300.0,
                help="focus measures above this value will be considered 'clear'")
args = vars(ap.parse_args())

# open files for logging
blurry_file = open("blurry.txt", "w")
other_file = open("other.txt", "w")
clear_file = open("clear.txt", "w")

# loop over the input images
for imagePath in paths.list_images(args["images"]):
    # load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
	# method
    filename = os.path.basename(imagePath)

    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)

    # if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
    if fm < args["threshold"]:
        text = "Blurry"
        blurry_file.write(filename + "\n")
    elif fm < args["clear_threshold"]:
        text = "Other"
        other_file.write(filename + "\n")
    else:
        text = "Clear"
        clear_file.write(filename + "\n")


    # Show the image - can be commented out. 
    # If you want to change to a better threshold, uncomment it and test on a smaller image set. 
    cv2.putText(image, "{} : {:.2f}".format(text, fm), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    key = cv2.waitKey(0)

