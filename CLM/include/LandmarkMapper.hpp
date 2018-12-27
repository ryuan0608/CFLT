//
//  LandmarkMapper.hpp
//  VSARDemo
//
//  Created by RongYuan on 9/3/18.
//  Copyright © 2018 VIPS. All rights reserved.
//

#ifndef LandmarkMapper_hpp
#define LandmarkMapper_hpp

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

extern const vector<vector<int>> LMK106_TO_LMK76;

extern const vector<vector<int>> LMK106_TO_LMK68;

extern const vector<vector<int>> LMK106_TO_LMK5;

cv::Mat mapFrom106Lmks(const cv::Mat &lmk106, const int n);

vector<cv::Point2f> mapFrom106Lmks(const vector<cv::Point2f> &lmk106, const int n);

#endif /* LandmarkMapper_hpp */
