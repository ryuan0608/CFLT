//
//  LandmarkLocaliser.hpp
//
//  Main entry of CLM algorithm.
//  Performs Detection-by-tracking as well.
//
//  Created by Rong Yuan on 7/25/18.
//

#ifndef LandmarkLocaliser_hpp
#define LandmarkLocaliser_hpp

#include <stdio.h>
#include "CLM.hpp"
#include "PDM.hpp"
#include "CLMConfig.h"

using namespace std;


// Performs Lucas-Kanade-Tracker for Optical flow tracking
// Performs Template tracking
class LandmarkTracker
{
public:
    LandmarkTracker();
    
    // Do optical flow or template tracking
    int track(const cv::Mat &grayImg, vector<cv::Point> &trackedLmks);
    
    // Observe current landmark locations
    // and use them as optical flow starting point on next frame
    // Set initial to true only if no previous track exists
    void observe(const cv::Mat &grayImg, const vector<cv::Point> lmks2d, bool initial);
    
private:
    // No matter what tracking method is used,
    // reinitialize every few frames to reduce drift
    int reinit_count = 0;
    
    // ** KALMAN FILTER **
    cv::KalmanFilter KF;
    cv::Mat state, processNoise, measurement;
    
    // ** OPTICAL FLOW **
    cv::Size mWinSize;
    cv::TermCriteria termcrit;
    cv::Mat prevTrackedImg, currTrackedImg;
    cv::Mat prevTrackedLmks, currTrackedLmks;
    
    void trackByOpticalFlow(const cv::Mat &prevImg, const cv::Mat &currImg, const cv::Mat &prevPts, cv::Mat &currPts, const int downsample = 1);
    
    // ** TEMPLATE TRACKING **
    bool trackByTemplateMatching(const cv::Mat &prevImg, const cv::Mat &currImg, const cv::Mat &prevPts, cv::Mat &currPts, const int downsample = 1);
    
    // ** BOUNDING BOX TRACKING **
    cv::Rect prevBbx, currBbx;
    void calcBoundingBox(cv::Rect &bbx, const vector<cv::Point> lmks);
    
};

// CLM entry
class LandmarkLocaliser
{
public:
    LandmarkLocaliser();
    LandmarkLocaliser(const string model_file_path);
    // pass initial landmark estiamtes in
    // and received 106 landmarks as output
    template<typename Callable>
    bool LocaliseLandmarks(cv::Mat &rgbImg, Callable detectLandmarks, vector<cv::Point> &lmks2d)
    {
        // 1. try tracking first
        cv::Mat grayImg;
        cv::cvtColor(rgbImg, grayImg, CV_RGB2GRAY);
        
        int flag = landmarkTracker->track(grayImg, lmks2d);
        
        // 2. /* DETECTION-BY-TRACKING */
        // decide whether to use tf face detection or not
        // depending on optical flow distance
        // No Previous Track        : USE TF
        // Reinitialization Required: USE TF
        // Small motion:            : USE OPTICAL FLOW
        bool faceDetected = false;
        switch (flag) {
            case NO_PREV_TRACK:
                lmks2d = detectLandmarks(rgbImg);
                if(!lmks2d.empty())
                {
                    clm.init(lmks2d);
                    faceDetected = clm.Localise(grayImg, clmParameters, lmks2d);
                    lmks2d.clear();
                    for(int i = 0; i < clm.detected_landmarks.rows; i++)
                        lmks2d.push_back(cv::Point(clm.detected_landmarks.row(i)));
                    landmarkTracker->observe(grayImg, lmks2d, true);
                }
                break;
            case REDETECTION:
                lmks2d = detectLandmarks(rgbImg);
                if(!lmks2d.empty())
                {
                    clm.init(lmks2d);
                    faceDetected = clm.Localise(grayImg, clmParameters, lmks2d);
                    lmks2d.clear();
                    for(int i = 0; i < clm.detected_landmarks.rows; i++)
                        lmks2d.push_back(cv::Point(clm.detected_landmarks.row(i)));
                    landmarkTracker->observe(grayImg, lmks2d, false);
                }
                break;
            case OPTICAL_FLOW:
                if(!lmks2d.empty())
                    landmarkTracker->observe(grayImg, lmks2d, false);
                break;
            default:
                break;
        }
        
        return faceDetected;
    }
    
private:
    // ** CLM **
    bool faceExists = false;
    CLM clm;
    CLMParameters clmParameters;
    LandmarkTracker *landmarkTracker;
};


#endif /* LandmarkLocaliser_hpp */
