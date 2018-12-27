//
//  CLM_utils.hpp
//  Pods
//
//  Created by RongYuan on 7/20/18.
//

#ifndef CLM_utils_hpp
#define CLM_utils_hpp

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;

extern string OPEN_FACE_DIR;

vector<string> getDirectoryFiles(const string& dir);

// Skipping lines that start with # (together with empty lines)
void SkipComments(std::ifstream& stream);

// Reading in a matrix from a stream
void ReadMat(std::ifstream& stream, cv::Mat &output_mat);

void ReadMatBin(std::ifstream& stream, cv::Mat &output_mat);

cv::Mat AlignShapesKabsch2D_f(const cv::Mat_<float>& align_from, const cv::Mat_<float>& align_to);

void crossCorr_m(const cv::Mat& img, cv::Mat_<double>& img_dft, const cv::Mat_<float>& _templ, map<int, cv::Mat_<double> >& _templ_dfts, cv::Mat& corr);

void matchTemplate_m(const cv::Mat& input_img, cv::Mat_<double>& img_dft, cv::Mat& _integral_img, cv::Mat& _integral_img_sq, const cv::Mat_<float>&  templ, map<int, cv::Mat_<double> >& templ_dfts, cv::Mat& result, int method);

cv::Mat AlignShapesWithScale_f(cv::Mat& src, cv::Mat dst);

void Grad(const cv::Mat& im, cv::Mat& grad);

cv::Vec3f RotationMatrix2Euler(const cv::Mat& rotation_matrix);

cv::Mat intrinsic();

#endif /* CLM_utils.hpp */
