#include <vector>
#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "ldmarkmodel.h"

#include "LandmarkLocaliser.hpp"

using namespace std;
using namespace cv;


const string FACE_MODEL_PATH = "/Users/orangechicken/Documents/Developer/HyperLandmarks/HyperLandmarks/model/haar_facedetection.xml";
const string LMK_MODEL_PATH = "/Users/orangechicken/Documents/Developer/HyperLandmarks/HyperLandmarks/model/landmark-model.bin";
const string CLM_MODEL_PATH = "/Users/orangechicken/Documents/Developer/HyperLandmarks/CLM/model/vip_106.txt";

int main()
{

   ldmarkmodel modelt(FACE_MODEL_PATH);
   std::string modelFilePath = LMK_MODEL_PATH;

   if(!load_ldmarkmodel(LMK_MODEL_PATH, modelt)){
       std::cout << "Model Opening Failed." << std::endl;
       std::cin >> modelFilePath;
   }

   cv::VideoCapture mCamera(0);
   if(!mCamera.isOpened()){
       std::cout << "Camera Opening Failed..." << std::endl;
       return 0;
   }
   cv::Mat rgbImg;
   std::vector<cv::Mat> currentShape(MAX_FACE_NUM);

   LandmarkLocaliser localiser(CLM_MODEL_PATH);
    
vector<cv::Point> lmks;
   while(1){
       mCamera >> rgbImg;
       cv::resize(rgbImg, rgbImg, rgbImg.size() / 2);
       cv::flip(rgbImg, rgbImg, 1);
       
       localiser.LocaliseLandmarks(rgbImg, std::bind(&ldmarkmodel::track, &modelt, std::placeholders::_1), lmks);

       for(cv::Point p: lmks)
           cv::circle(rgbImg, p, 2, cv::Scalar(255, 0, 255));
       
       cv::imshow("Camera", rgbImg);
       if(27 == cv::waitKey(5)){
           mCamera.release();
           cv::destroyAllWindows();
           break;
       }
   }

   return 0;
}






















