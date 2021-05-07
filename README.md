# License-Plate-Recognition
License Plate Recognition using OpenCV and TesseractOCR

# General Idea
It is to design a license plate recognition system that could process and identify the 20 sample images under different angle and light condition.

There are various image processing techniques are utilized in this project.

* The original image will be converted to gray scale.
* Equalized Histogram will be performed to brighten the image and adjust the contrast.
* The image will be blurred to remove high-frequency content like edges of the image to make it smooth.
* The Sobel edge detection will be performed to extract the edges of the image.
* The dilation will be applied to isolate the individual elements and remove noise.
* The OTSU thresholding method is used to convert the gray image to binary image to find the region of interest of the images.
* The segmented image will be passed to the Tesseract OCR for character recognition.

# Tech/Framework Used
1. OpenCV
2. TesseractOCR

## Languages
1. C++

### built with
1. Visual Studio 2017

# Screenshots
## Original Image
![image](https://user-images.githubusercontent.com/63278063/117431277-a16b9880-af5b-11eb-9998-9bd1c10a53ae.png)

