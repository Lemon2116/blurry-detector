# Blurry Detector
```blurry-detector``` contains a blurry detector that detects blurriness of images, dividing them into "clear", "blurry", and "other" categories. It used the variance of the Laplacian method to calculate the change of intensity over the image. 

## Directory Structure
```blurry-detector/
├── img/                                           # Sample blurry/clear/other images
├── detectors/                                          
|   ├──detect_blur.py                              # General global detector
│   └──detect_blur_enhanced.py                     # Enhanced detector which can choose from different modes
└── README.md
```

## Application
* Database cleaning and Image Selection.
* Image Categorization.

## Acknowledgement
* [Blur detection with OpenCV](https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/) for algorithm inspiration.
