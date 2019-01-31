# CLM-based Facial Landmark Tracker (CFLT)
This is a facial landmark tracker dependent on [SDM](https://github.com/zeusees/HyperLandmark) and [CLM](https://github.com/TadasBaltrusaitis/OpenFace). Primarily it is supposed to be part of VIP ARSDK and serves as a facial landmark refinement upon MTCNN detection result. However I includ a simple open-source SDM implementation here only to make it a somewhat complete stand-alone module. 

## 1. Overview
?
CLM-based Facial Landmark Tracker consists of two major modules with one's result feeding into the other as input. Here SDM gives an intial esimate of 68 landmarks when tracking is not available. Have the estimate given, CLM computes a higher resolution result as well as a track over frames. After an interval of time, or whenever a track is not sufficiently accurate, SDM comes in again to reinitialize.  

## 2. How to run the code?
A CMake file is available under the root directory. Make sure you have OpenCV>=3.2 installed and you are good.  

## 3. How to replace SDM with other detectors?
In CLMFaceTest.cpp you will find these two lines as below.  
```c++
LandmarkLocaliser localiser(CLM_MODEL_PATH);
.
.
.
localiser.LocaliseLandmarks(rgbImg, std::bind(&ldmarkmodel::track, &sdm, std::placeholders::_1), lmks);  
```

In CLM/LandmarkLocaliser.hpp you will find the declaration of LocaliseLandmarks as below.
```c++
template<typename Callable>
    bool LocaliseLandmarks(cv::Mat &rgbImg, Callable detectLandmarks, vector<cv::Point> &lmks2d)
```

Essentially LocaliseLandmarks is the only entrance of the CFLT, and it takes input camera image, a function pointer, as well as a placeholder for output landmarks. Therefore, to replace sdm with other detectors, simple construct your own "detectLandmarks" function and pass it to LocaliseLandmarks as a pointer. Note that your own detector function should take a cv::Mat as input and return a vector of cv::Point as output. See the sample code below for details.  

```c++
// Input:  rgb image [camera input]
// Output: 106 landmarks in the format of cv::Point
std::vector<cv::Point> track(const cv::Mat& src);
```

## 4. Who can I turn to shall I have any further questions?
Contact me (Rong Yuan) via email: yuanrong0608@gmail.com
