//
//  PDM.hpp
//
//  Created by Rong Yuan on 7/23/18.
//

#ifndef PDM_hpp
#define PDM_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include <set>
#include <opencv2/opencv.hpp>

#include "CLMConfig.h"

using namespace std;

class PDM
{
public:
    PDM();
    
    bool Read(string location);
    cv::Mat calc2DShape() const;
    
    // public setters and getters
    int nDim() const;
    int nPts() const;
    cv::Mat mean() const;
    cv::Mat evs() const;
    cv::Mat bases() const;
    cv::Vec3d getRvec() const;
    cv::Vec3d getTvec() const;
    void setRvec(const cv::Vec3d &_rvec);
    void setTvec(const cv::Vec3d &_tvec); 
    void getParams(cv::Mat &output) const;
    void setParams(const cv::Mat &_params);
    void getShape(cv::Mat &output) const;
    
    

    // virtual functions
    // implementation details vary from parametric models
    void updateShape();
    // 2NxK jacobian matrix w.r.t. pca model weights
    void jacobianWeights(cv::Mat &output) const;
    // 2Nx6 jacobian matrix w.r.t. 6 pose parameters(alpha, beta, theta, Tx, Ty, Tz)
    void jacobianRT(cv::Mat &output) const;
    // 2Nx(K+6) jacobian matrix combining both weights and pose parameters
    void jacobian(cv::Mat &output) const;
    // 2Nx1 residual vector
    void residual(cv::Mat &output) const;
    float err() const;
    void calcParams(vector<cv::Point2f> lmks2d);

private:
    int N, K;
    
    cv::Mat B; // bases reaylly used in equations (3MxN)
    cv::Mat params; // current weights for active shape (Mx1)
    cv::Mat M; // mean shape(3xN)
    cv::Mat E; // eigenvalues (Mx1)
    
    cv::Mat W; // detected landmarks
    
    cv::Mat shape; // shape is defined as the 3d point cloud without rotation/translation/scaling
    
    cv::Vec3d rvec, tvec;
    cv::Mat R33, T3;
    int timestamp = 0;
};

#endif /* PDM_hpp */
