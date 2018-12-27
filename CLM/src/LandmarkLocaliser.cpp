//
//  LandmarkDetector.cpp
//
//  Main entry of CLM algorithm.
//  Performs Detection-by-tracking as well.
//
//  Created by Rong Yuan on 7/25/18.
//

#include "LandmarkLocaliser.hpp"
#include "LandmarkMapper.hpp"
#include "CLM_utils.hpp"

LandmarkLocaliser::LandmarkLocaliser()
{
    // ** CLM **
    clm = CLM(OPEN_FACE_DIR + "vip_106.txt");
    clmParameters = CLMParameters();
    landmarkTracker = new LandmarkTracker();
}
LandmarkLocaliser::LandmarkLocaliser(const string model_file_path)
{
    clm = CLM(model_file_path);
    clmParameters = CLMParameters();
    landmarkTracker = new LandmarkTracker();
}


LandmarkTracker::LandmarkTracker()
{
    // ** KALMAN FILTER **
//    const int nStates = 68 * 4;
//    const int nMeasurements = 68 * 2;
//    KF = cv::KalmanFilter(nStates, nMeasurements, 0);
//    state = cv::Mat(nStates, 1, CV_32F);
//    processNoise = cv::Mat(nStates, 1, CV_32F);
//    measurement = cv::Mat::zeros(nMeasurements, 1, CV_32F);
//    // generate transition matrix
//    KF.transitionMatrix = cv::Mat::zeros(nStates, nStates, CV_32F);
//    for(int i = 0; i < nStates; i++)
//        for(int j = 0; j < nStates; j++)
//            if(i == j || j - i == nMeasurements)
//                KF.transitionMatrix.at<float>(i, j) = 1.0f;
//    cv::setIdentity(KF.measurementMatrix);
//    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-1));
//    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
//    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(0.1));
    
    // ** OPTICAL FLOW **
    mWinSize = cv::Size(11, 11);
    termcrit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
}

// Tracking algorithm in steps:
// 1. If no previous track exists, then initialize with tf face result
// 2. Otherwise, do Optical flow first
// 3. Decide which method to use depending on optical flow distance
//    If reinitialization is required, use tf result            [Slow]
//    If motion is very large, reinitialize with tf face result [Slow]
//    If motion is very small, use optical flow result          [Fast]
//    Otherwise, use template matching result                   [Fast]
int LandmarkTracker::track(const cv::Mat &grayImg, vector<cv::Point> &trackedLmks)
{
    if(!prevTrackedLmks.empty())
    {
        if(++reinit_count % EVERY_TRACKING_REINIT > 0) // do tracking
        {
            int left = grayImg.cols, right = -1;
            for(int i = 0; i < prevTrackedLmks.rows; i++)
            {
                float x = prevTrackedLmks.at<float>(i, 0);
                //                float y = prevTrackedLmks.at<float>(i, 1);
                left = min(left, int(x));
                right = max(right, int(x));
            }
            int width = right - left;
            
            trackByOpticalFlow(prevTrackedImg, grayImg, prevTrackedLmks, currTrackedLmks, TRACKING_DOWNSAMPLE_RATE);
            int n = prevTrackedLmks.rows;
            double minVal, maxVal;
            cv::minMaxLoc(prevTrackedLmks.reshape(1, n*2) - currTrackedLmks.reshape(1, n*2), &minVal, &maxVal);
            if(maxVal * TRACKING_DOWNSAMPLE_RATE >= float(MAX_TRACKING_DIST) * width / 200.0)
            {
                reinit_count = 0;
                return REDETECTION;
            }
            trackedLmks.clear();
            for(int i = 0; i < currTrackedLmks.rows; i++)
                trackedLmks.push_back(cv::Point(currTrackedLmks.row(i)));
            grayImg.copyTo(prevTrackedImg);
            return OPTICAL_FLOW;
        }
        else // reinit
        {
            reinit_count = 0;
            return REDETECTION;
        }
    }
    return NO_PREV_TRACK;
}

// Update landmark locations
// to prepare for tracking [both optical flow and template] in next frame
void LandmarkTracker::observe(const cv::Mat &grayImg, const vector<cv::Point> lmks2d, bool initial)
{
    if(initial)
    {
//        KF.statePre.setTo(0.0);
//        for(int i = 0; i < 68; i++)
//        {
//            KF.statePre.at<float>(2*i) = float(lmks2d[i].x);
//            KF.statePre.at<float>(2*i+1) = float(lmks2d[i].y);
//        }
        int n = lmks2d.size();
        prevTrackedLmks = cv::Mat(n, 2, CV_32F);
        calcBoundingBox(prevBbx, lmks2d);
    }
    // update track
    for(int i = 0; i < int(lmks2d.size()); i++)
    {
        prevTrackedLmks.at<float>(i, 0) = float(lmks2d[i].x);
        prevTrackedLmks.at<float>(i, 1) = float(lmks2d[i].y);
    }
    grayImg.copyTo(prevTrackedImg);
}

void LandmarkTracker::trackByOpticalFlow(const cv::Mat &prevImg, const cv::Mat &currImg, const cv::Mat &prevPts, cv::Mat &currPts, const int downsample)
{
    std::vector<uchar> status;
    std::vector<float> err;
    int maxLevel = 1;
    if(downsample > 1)
    {
        cv::Mat dsPrevPts, dsCurrPts;
        cv::Mat dsPrevImg, dsCurrImg;
        dsPrevPts = prevPts / downsample;
        cv::resize(prevImg, dsPrevImg, cv::Size(prevImg.cols / downsample, prevImg.rows / downsample));
        cv::resize(currImg, dsCurrImg, cv::Size(currImg.cols / downsample, currImg.rows / downsample));
        cv::calcOpticalFlowPyrLK(dsPrevImg, dsCurrImg, dsPrevPts, dsCurrPts, status, err, mWinSize, maxLevel,termcrit);
        currPts = dsCurrPts * downsample;
    }
    else
    {
        cv::calcOpticalFlowPyrLK(prevImg, currImg, prevPts, currPts, status, err, mWinSize, maxLevel,termcrit);
    }
}

bool LandmarkTracker::trackByTemplateMatching(const cv::Mat &prevImg, const cv::Mat &currImg, const cv::Mat &prevPts, cv::Mat &currPts, const int downsample)
{
    // 1. downsample
    cv::Mat dsPrevPts, dsCurrPts;
    cv::Mat dsPrevImg, dsCurrImg;
    dsPrevPts = prevPts / downsample;
    cv::resize(prevImg, dsPrevImg, cv::Size(prevImg.cols / downsample, prevImg.rows / downsample));
    cv::resize(currImg, dsCurrImg, cv::Size(currImg.cols / downsample, currImg.rows / downsample));
    
    // 2. extract bounding boxes and define roi
    float min_x = dsPrevImg.rows + dsPrevImg.cols, min_y =  dsPrevImg.rows + dsPrevImg.cols;
    float max_x = -1, max_y = -1;
    for(int i = 0; i < prevPts.rows; i++)
    {
        float x = dsPrevPts.at<float>(i, 0);
        float y = dsPrevPts.at<float>(i, 1);
        if(x > max_x)
            max_x = x;
        if(x < min_x)
            min_x = x;
        if(y > max_y)
            max_y = y;
        if(y < min_y)
            min_y = y;
    }
    float width = max_x - min_x, height = max_y - min_y;
    cv::Rect bbx(min_x, min_y, width, height);
    bbx = bbx & cv::Rect(0, 0, dsCurrImg.cols, dsCurrImg.rows);
    cv::Rect roi(bbx.x - width/2, bbx.y - height/2, 2*width, 2*height);
    roi = roi & cv::Rect(0, 0, dsCurrImg.cols, dsCurrImg.rows);
    
    cv::Mat templ = dsPrevImg(bbx);
    cv::Mat searchArea = dsCurrImg(roi);
    
    // 3. do actual template matching
    cv::Mat result;
    cv::matchTemplate(searchArea, templ, result, CV_TM_CCOEFF_NORMED);
    // 4. find max response location
    int maxResponse[2];
    double minVal, maxVal;
    cv::minMaxIdx(result, &minVal, &maxVal, NULL, maxResponse);
    if(maxVal < 2 * minVal)
        return false;
    int upsample = downsample;
    float shift_x = (maxResponse[1]+roi.x-bbx.x)*upsample;
    float shift_y = (maxResponse[0]+roi.y-bbx.y)*upsample;
    
    // 5. generate tracked landmark positions
    prevPts.copyTo(currPts);
    for(int i = 0; i < currPts.rows; i++)
    {
        currPts.at<float>(i, 0) += shift_x;
        currPts.at<float>(i, 1) += shift_y;
    }
    return true;
}

void LandmarkTracker::calcBoundingBox(cv::Rect &bbx, const vector<cv::Point> lmks)
{
    // Get the width of expected shape
    double min_x, max_x, min_y, max_y;
    min_x = lmks[0].x;
    max_x = lmks[0].x;
    min_y = lmks[0].y;
    max_y = lmks[0].y;
    for(int i = 1; i < lmks.size(); i++)
    {
        if(lmks[i].x > max_x)
            max_x = lmks[i].x;
        if(lmks[i].x < min_x)
            min_x = lmks[i].x;
        if(lmks[i].y > max_y)
            max_y = lmks[i].y;
        if(lmks[i].y < min_y)
            min_y = lmks[i].y;
    }
    
    float width = abs(min_x - max_x);
    float height = abs(min_y - max_y);
    
    bbx = cv::Rect(min_x, min_y, width, height);
}


