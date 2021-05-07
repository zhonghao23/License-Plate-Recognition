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

## Grayscale Image
![image](https://user-images.githubusercontent.com/63278063/117431364-b811ef80-af5b-11eb-8fa7-d30757e8331f.png)

## Equalized Histogram
![image](https://user-images.githubusercontent.com/63278063/117431391-c233ee00-af5b-11eb-9d51-1415592c117d.png)

## Blur
![image](https://user-images.githubusercontent.com/63278063/117431415-ca8c2900-af5b-11eb-9508-b0cb83549f4f.png)

## Sobel Edge Detection
![image](https://user-images.githubusercontent.com/63278063/117431446-d2e46400-af5b-11eb-8607-e4bf1172aea9.png)

## Dilation
![image](https://user-images.githubusercontent.com/63278063/117431557-f6a7aa00-af5b-11eb-9ed3-f911e29657e6.png)

## OTSU Binarized Segmented Plate
![image](https://user-images.githubusercontent.com/63278063/117431602-045d2f80-af5c-11eb-81f1-bbe27bea68d0.png)

## Noise Removal & Dilated Plate
![image](https://user-images.githubusercontent.com/63278063/117431683-18089600-af5c-11eb-84b7-058d651551ef.png)

## Final Extracted Plate
![image](https://user-images.githubusercontent.com/63278063/117431792-35d5fb00-af5c-11eb-8773-4035c25cc62a.png)

## Result
![image](https://user-images.githubusercontent.com/63278063/117431955-61f17c00-af5c-11eb-87dc-a99cadf1a53f.png)

# Verdict
The accuracy of the license plate recognition system could hit 85%. Some of the words like V will be detected as Y.
