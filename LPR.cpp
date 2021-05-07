//#include "pch.h"
#include <iostream>
#include <string>
#include "core/core.hpp"
#include "highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#pragma warning(disable : 4996)
#include "opencv2/imgproc.hpp"
#include <baseapi.h>
#include <allheaders.h>

using namespace std;
using namespace cv;
void plateRecognition(Mat plate[], int count, Mat image, int th, int otsuTH);
void ocrForPlate(Mat plate, int count, int th, int segmentCount);
int segmentByChar(Mat plate, int count, Mat image, int th, int otsuTH);

Mat convertToGrey(Mat RGBimage)
{
	Mat Grey = Mat::zeros(RGBimage.size(), CV_8UC1);
	for (int i = 0; i < RGBimage.rows; i++) {
		for (int j = 0; j < RGBimage.cols * 3; j = j + 3) {	//each channel of RGB image got 3 channels
			Grey.at<uchar>(i, j / 3) = (RGBimage.at<uchar>(i, j) + RGBimage.at<uchar>(i, j + 1) + RGBimage.at<uchar>(i, j + 2)) / 3;
		}
	}
	return Grey;
}

Mat EqualizeHistogram(Mat Grey) {
	int count[256] = { 0 };
	float prob[256] = { 0.0 };
	float accProb[256] = { 0.0 };
	int newPixel[256] = { 0 };

	Mat EqualizedImage = Mat::zeros(Grey.size(), CV_8UC1);

	//Calculate the Count
	for (int i = 0; i < Grey.rows; i++) {
		for (int j = 0; j < Grey.cols; j++) {
			count[Grey.at<uchar>(i, j)]++;
		}
	}
	//Calculate the Probability
	int totalPixel = Grey.rows * Grey.cols;
	for (int i = 0; i < 256; i++) {
		prob[i] = (float)((float)count[i] / (float)totalPixel);
	}
	//Calculate the Accumulative Probability
	accProb[0] = prob[0];
	for (int i = 1; i < 256; i++) {
		accProb[i] = accProb[i - 1] + prob[i];
	}
	//Calculate the New Pixel [ (G-1) * accProb ]
	for (int i = 0; i < 256; i++) {
		newPixel[i] = accProb[i] * 255;
	}
	//Replacing New Pixel
	for (int i = 0; i < Grey.rows; i++) {
		for (int j = 0; j < Grey.cols; j++) {
			EqualizedImage.at<uchar>(i, j) = newPixel[Grey.at<uchar>(i, j)];
		}
	}
	return EqualizedImage;
}

Mat Blur(Mat Grey, int nb) {
	Mat BlurImage = Mat::zeros(Grey.size(), CV_8UC1);
	int windowsize = (2 * nb + 1) * (2 * nb + 1);
	for (int i = nb; i < Grey.rows - nb; i++) {
		for (int j = nb; j < Grey.cols - nb; j++) {
			int sum = 0;
			for (int ii = -nb; ii <= nb; ii++) {
				for (int jj = -nb; jj <= nb; jj++) {
					sum += Grey.at<uchar>(i + ii, j + jj);
				}
			}
			BlurImage.at<uchar>(i, j) = sum / windowsize;
		}
	}
	return BlurImage;
}

Mat EdgeDetection(Mat Grey, int nb, int th) {
	Mat EdgeDetectedImage = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = nb; i < Grey.rows - nb; i++) {
		for (int j = nb; j < Grey.cols - nb; j++) {
			float avgL = (Grey.at<uchar>(i - 1, j - 1) + Grey.at<uchar>(i, j - 1) + Grey.at<uchar>(i + 1, j - 1)) / 3;
			float avgR = (Grey.at<uchar>(i - 1, j + 1) + Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i + 1, j + 1)) / 3;
			if (abs(avgL - avgR) > th) {
				EdgeDetectedImage.at<uchar>(i, j) = 255;
			}
		}
	}
	return EdgeDetectedImage;
}

Mat Dilation(Mat Edge, int nb) {
	Mat DilationImage = Mat::zeros(Edge.size(), CV_8UC1);
	int windowsize = (2 * nb + 1) * (2 * nb + 1);
	for (int i = nb; i < Edge.rows - nb; i++) {
		for (int j = nb; j < Edge.cols - nb; j++) {
			for (int ii = -nb; ii <= nb; ii++) {
				for (int jj = -nb; jj <= nb; jj++) {
					if (Edge.at<uchar>(i + ii, j + jj) == 255) {
						DilationImage.at<uchar>(i, j) = 255;
					}
				}
			}
		}
	}
	return DilationImage;
}


int otsu(Mat plate) {
	int count[256] = { 0 };
	float prob[256] = { 0.0 };
	int newPixel[256] = { 0 };
	float meu[256] = { 0.0 };
	float sigma[256] = { 0.0 };
	float teta[256] = { 0.0 };
	int threshold;

	//Calculate the Count
	for (int i = 0; i < plate.rows; i++) {
		for (int j = 0; j < plate.cols; j++) {
			count[plate.at<uchar>(i, j)]++;
		}
	}
	//Calculate the Probability
	int totalPixel = plate.rows * plate.cols;
	for (int i = 0; i < 256; i++) {
		prob[i] = (float)((float)count[i] / (float)totalPixel);
	}
	//Calculate the Accumulative Probability
	teta[0] = prob[0];
	for (int i = 1; i < 256; i++) {
		teta[i] = teta[i - 1] + prob[i];
	}

	meu[0] = prob[0];
	for (int i = 1; i < 256; i++) {
		meu[i] = meu[i - 1] + i * prob[i];
	}

	for (int i = 0; i < 256; i++) {
		sigma[i] = ((meu[255] * teta[i] - meu[i]) * (meu[255] * teta[i] - meu[i])) / (teta[i] * (1 - teta[i]));
	}


	threshold = 0;
	// find i which has the max sigma
	for (int i = 1; i < 256; i++) {
		if (sigma[i] > sigma[i - 1]) {
			threshold = i;
		}
	}
	//cout << threshold;
	return threshold;
}

Mat convertToBinary(Mat Grey, int Th)
{
	Mat binary = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 0; i < Grey.rows; i++) {
		for (int j = 0; j < Grey.cols; j++) {
			if (Grey.at<uchar>(i, j) >= Th) {
				binary.at<uchar>(i, j) = 255;
			}
		}
	}
	return binary;
}

Mat InitialImage;
bool foundPlate;

void imageProcessing(Mat image, int th) { //dilation 3, blur 1, th45
	foundPlate = false;
	imshow("RBG Image", image);
	waitKey();
	destroyWindow("RBG Image");
	cout << "Th" << th << endl;
	Mat Grey = convertToGrey(image);
	Mat EqualizedImage = EqualizeHistogram(Grey);
	Mat BlurImage = Blur(EqualizedImage, 1);
	Mat EdgeDetectedImage = EdgeDetection(BlurImage, 1, th);
	Mat DilatedImage = Dilation(EdgeDetectedImage, 3);
	imshow("Dilated Image", DilatedImage);
	waitKey();
	destroyWindow("Dilated Image");

	Mat Blob = DilatedImage.clone();
	vector<vector<Point>> contours1;
	vector<Vec4i> hierarchy1;
	findContours(DilatedImage, contours1, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

	Rect rect;
	Scalar black = CV_RGB(0, 0, 0);
	Mat plate[3];
	int count = 0;
	double area, rectWidth, rectHeight, rectArea, check;
	for (size_t j = 0; j < contours1.size(); j++)
	{
		rect = boundingRect(contours1[j]);
		rectWidth = rect.width;
		rectHeight = rect.height;
		rectArea = rectWidth * rectHeight;
		area = contourArea(contours1[j]); //get area of segment
		check = area / rectArea;
		if (rect.width < image.cols * 0.06 || rect.width > image.cols * 0.2 || rect.height < image.rows * 0.03 || rect.height > image.rows * 0.10 ||
			(rect.height / rect.width) > 2 || check < 0.6 ||
			rect.x < image.cols * 0.1 || rect.x > image.cols * 0.85 || rect.y < image.rows * 0.1 || rect.y > image.rows * 0.9)
		{
			drawContours(Blob, contours1, j, black, -1, 8, hierarchy1);
		}
		else
		{
			plate[count] = Grey(rect);
			count++;
		}
	}
	bool blackImage = true;
	for (int i = 0; i < Blob.rows; i++) {
		for (int j = 0; j < Blob.cols; j++) {
			if (Blob.at<uchar>(i, j) != 0) {
				blackImage = false;
			}
		}
	}
	if (blackImage == true) {
		th = th - 10;
		if (th >= 20) {
			imageProcessing(image, th);
		}
		else {
			th = 50;
			imageProcessing(image, th);
		}
	}
	else {
		imshow("Filtered Image", Blob);
		waitKey();
		destroyWindow("Filtered Image");
		plateRecognition(plate, count, image, th, 25);
	}
}

Mat plateNum1[20];
Mat plateNum2[20];
Mat plateNum3[20];

void plateRecognition(Mat plate[], int count, Mat image, int th, int otsuTH) //OCR IN PLATE & CHARACTERS
{
	for (int i = 0; i <= count - 1; i++) {
		if (foundPlate == true) {
			break;
		}
		imshow("Plate", plate[i]);	//DISPLAYING THE WHOLE PLATE IN GREY
		waitKey();
		destroyWindow("Plate");

		if (otsuTH == 25) {
			resize(plate[i], plate[i], cv::Size(), 3, 3); //ori 3
		}

		int threshold = otsu(plate[i]);	//LOOK FOR THE THRESHOLD FOR BINARIZATION
		threshold = threshold + otsuTH;

		Mat binaryPlate = convertToBinary(plate[i], threshold);	//BINARIZE GREY PLATE

		imshow("Binary Plate", binaryPlate);	//DISPLAYING THE WHOLE BINARIZED PLATE
		waitKey();
		destroyWindow("Binary Plate");

		vector<vector<Point>> contours2;
		vector<Vec4i> hierarchy2;
		findContours(binaryPlate, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));	//FIND CHARACTERS IN PLATE
		Rect rect;
		Scalar black = CV_RGB(0, 0, 0);

		if (!contours2.empty()) {
			for (size_t j = 0; j < contours2.size(); j++) {
				rect = boundingRect(contours2[j]);
				if (rect.height < binaryPlate.rows * 0.3 || rect.width > binaryPlate.cols * 0.8 || //ori 0.5
					rect.x < binaryPlate.cols * 0.05 || rect.x > binaryPlate.cols * 0.95 || rect.y < binaryPlate.rows * 0.03 || rect.y > binaryPlate.rows * 0.97) {	//CLEARING NOISES IN A WHOLE PLATE
					drawContours(binaryPlate, contours2, j, black, -1, 8, hierarchy2);
				}
			}

			imshow("Binary Plate", binaryPlate);	//DISPLAYING THE WHOLE BINARIZED PLATE
			waitKey();
			destroyWindow("Binary Plate");
			Mat Blob = binaryPlate.clone();
			if (otsuTH == 25) {	//RUN WITH OTSU 25
				int count1 = segmentByChar(Blob, count, image, th, otsuTH);
				ocrForPlate(binaryPlate, 1, th, count1);
				otsuTH = 50;
				plateRecognition(plate, count, image, th, otsuTH);
			}
			else if (otsuTH == 50) {	//RUN WITH OTSU 50
				int count2 = segmentByChar(Blob, count, image, th, otsuTH);
				ocrForPlate(binaryPlate, 2, th, count2);
				otsuTH = 65;
				plateRecognition(plate, count, image, th, otsuTH);
			}
			else if (otsuTH == 65) {	//RUN WITH OTSU 65 
				int count3 = segmentByChar(Blob, count, image, th, otsuTH);
				ocrForPlate(binaryPlate, 3, th, count3);
			}
		}

	}
}

int segmentByChar(Mat plate, int count, Mat image, int th, int otsuTH) {
	int count1 = 0, count2 = 0, count3 = 0;

	vector<vector<Point>> contours2;
	vector<Vec4i> hierarchy2;
	findContours(plate, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));	//FIND CHARACTERS IN PLATE
	Rect rect;
	Scalar black = CV_RGB(0, 0, 0);

	if (!contours2.empty()) {
		for (size_t i = 0; i < contours2.size(); i++)
		{
			rect = boundingRect(contours2[i]);
			if (rect.height < plate.rows * 0.3 || rect.width > plate.cols * 0.7)
			{
				drawContours(plate, contours2, i, black, -1, 8, hierarchy2);
			}
			else
			{
				if (otsuTH == 25)
				{
					plateNum1[count1] = plate(rect);
					//imshow("PlateNumber", plateNum1[count1]);
					//waitKey();
					//destroyWindow("PlateNumber");
					count1++;
				}
				else if (otsuTH == 50)
				{
					plateNum1[count2] = plate(rect);
					//imshow("PlateNumber", plateNum1[count2]);
					//waitKey();
					//destroyWindow("PlateNumber");
					count2++;
				}
				else if (otsuTH == 65)
				{
					plateNum1[count1] = plate(rect);
					//imshow("PlateNumber", plateNum1[count3]);
					//waitKey();
					//destroyWindow("PlateNumber");
					count3++;
				}
			}
		}
	}
	if (otsuTH == 25) { return count1; }
	else if (otsuTH == 50) { return count2; }
	else if (otsuTH == 65) { return count3; }
}

Pix* mat8ToPix(cv::Mat* mat8)	//CONVERTING FROM MAT TO PIX IMAGE
{
	Pix* pixd = pixCreate(mat8->size().width, mat8->size().height, 8);
	for (int y = 0; y < mat8->rows; y++) {
		for (int x = 0; x < mat8->cols; x++) {
			pixSetPixel(pixd, x, y, (l_uint32)mat8->at<uchar>(y, x));
		}
	}
	return pixd;
}


bool checkPlate(char* outText, int segmentCount) {
	bool anyDigit = false;
	bool anyAlphabet = false;
	int i = 0;
	int wordCount = 0;
	if (!((outText[0] >= 'a' && outText[0] <= 'z') || (outText[0] >= 'A' || outText[0] <= 'Z'))) {	//FIRST CHARACTER MUST BE A-Z/a-z
		return false;
	}
	while (outText[i + 1] != NULL) { //have to include '\n' into wordCount
		if (i >= 10) {
			return false;
		}
		if ((outText[i] >= '0' && outText[i] <= '9')) {	//SEE IF ANY DIGIT INSIDE THE PLATE
			anyDigit = true;
		}
		if ((outText[i] >= 'a' && outText[i] <= 'z') || (outText[i] >= 'A' && outText[i] <= 'Z')) {	//SEE IF ANY ALPHABET INSIDE THE PLATE
			anyAlphabet = true;
		}
		if (outText[i] != '\n' && outText[i] != 32) //NOT EQUAL TO \n OR SPACE
		{
			if (!((outText[i] >= 'a' && outText[i] <= 'z') || (outText[i] >= 'A' && outText[i] <= 'Z') || (outText[i] >= '0' && outText[i] <= '9')))
			{
				return false;
			}
			else if ((outText[i] >= '0' && outText[i] <= '9'))
			{
				if ((outText[i + 1] >= 'a' && outText[i + 1] <= 'z') || (outText[i + 1] >= 'A' && outText[i + 1] <= 'Z'))
				{
					if ((outText[i + 2] >= '0' && outText[i + 2] <= '9'))
					{
						return false;
					}
				}
			}
			i++;
			wordCount++;
		}
		else if (outText[i] == 32)
		{
			if (!((outText[i + 1] == 32) || (outText[i + 1] >= '0' && outText[i + 1] <= '9'))) {
				return false;
			}
			i++;
			wordCount++;
		}
		else if (outText[i] == '\n') {
			i++;
			wordCount++;
		}
	}
	if (anyDigit == false || anyAlphabet == false) {	//IF THERE'S NO ANY DIGIT IN THE PLATE, FALSE
		return false;
	}
	if (i < 3) {
		return false;
	}
	if (wordCount < (segmentCount)) {	//TO CHECK IF THE WORD COUNT RECOGNIZED IS MORE THAN THE SEGMENTED CHARACTER COUNT
		return false;
	}
	return true;
}

char* outText1;
char* outText2;
char* outText3;
bool found1, found2, found3;
int outputNum = 1;

void ocrForPlate(Mat plate, int count, int th, int segmentCount) {	//OCR FOR WHOLE PLATE
	int countChar = 0;
	tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
	if (api->Init(NULL, "eng")) { 	//INITIALIZE TESSERACT-OCR WITH ENGLISH, WITHOUT SPECIFYING TESSDATA PATH
		fprintf(stderr, "Could not initialize tesseract.\n");
		exit(1);
	}
	Mat* plateChar1 = &plate;
	Pix* image = mat8ToPix(plateChar1);
	api->SetImage(image);
	api->SetSourceResolution(300);
	if (count == 1)	//OTSUTH25
	{
		outText1 = api->GetUTF8Text();
		printf("OCR output 1:\n%s", outText1);
		found1 = checkPlate(outText1, segmentCount);
	}
	else if (count == 2) { //OTSUTH50
		outText2 = api->GetUTF8Text();
		printf("OCR output 2:\n%s", outText2);
		found2 = checkPlate(outText2, segmentCount);
	}
	else if (count == 3) {	//OTSUTH65
		outText3 = api->GetUTF8Text();
		printf("OCR output 3:\n%s", outText3);
		found3 = checkPlate(outText3, segmentCount);
		if (found2 == true) {	//OTSU50 HAS THE BEST RESULT
			cout << outputNum << "." << endl;
			printf("FINAL OUTPUT:\n%s\n", outText2);
			outputNum++;
			foundPlate = true;
		}
		else if (found1 == true && found2 == false && found3 == false) {	//ONLY 1 CORRECT
			cout << outputNum << "." << endl;
			printf("FINAL OUTPUT:\n%s\n", outText1);
			outputNum++;
			foundPlate = true;
		}
		else if (found3 == true && found2 == false && found1 == false) {	//ONLY 1 CORRECT
			cout << outputNum << "." << endl;
			printf("FINAL OUTPUT:\n%s\n", outText3);
			outputNum++;
			foundPlate = true;
		}
		else if (found1 == true && found2 == false && found1 == true) {
			cout << outputNum << "." << endl;
			printf("FINAL OUTPUT:\n%s\n", outText3);
			outputNum++;
			foundPlate = true;
		}
		else if (found1 == false && found2 == false && found3 == false) {
			if (th < 10) {
				cout << "License Plate Could Not Be Recognized." << endl;
			}
			else {
				th = th - 10;
				imageProcessing(InitialImage, th);
			}
		}
		pixDestroy(&image);
	}
	api->End();
	delete api;	//DESTROY USED OBJECT & RELEASE MEMORY
}


int main()
{
	Mat images[20];
	string filename[20] = { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20" };
	string filepath1 = "C:\\Users\\mingh\\Desktop\\ISEPhoto\\";
	string filepath2 = ".jpg";
	for (int i = 0; i < 20; i++) {	//READING 20 IMAGES FROM FILE & SAVE IT INTO AN ARRAY
		images[i] = imread(filepath1 + filename[i] + filepath2);
	}
	//imageProcessing(images[5], 40); //ori 40
	//InitialImage = images[17];
	//imageProcessing(images[18], 40);
	for (int i = 0; i < 20; i++) {	//RUN THROUGH LPR FOR EVERY IMAGES
		InitialImage = images[i];
		imageProcessing(images[i], 40);
	}
}