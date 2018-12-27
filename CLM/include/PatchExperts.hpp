//
//  PatchExperts.hpp
//
//  Created by RongYuan on 7/20/18.
//

#ifndef PatchExperts_hpp
#define PatchExperts_hpp

#include <stdio.h>
#include <dirent.h>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include "PDM.hpp"
#include "SVR_patch_expert.hpp"
#include "CCNF_patch_expert.hpp"

#define N_SVR_SCALES 3
#define N_SVR_VIEWS 7

using namespace std;

class PatchExperts
{
public:
    PatchExperts(){;} // default constructor
    
    bool Read(vector<string> intensity_svr_expert_locations, vector<string> intensity_ccnf_expert_locations);
    bool Read_SVR_patch_experts(string svr_expert_location, std::vector<cv::Vec3d>& views, std::vector<cv::Mat>& visibility, std::vector<std::vector<Multi_SVR_PatchExpert> >& svr_templates, double& scale);
    bool Read_CCNF_patch_experts(string patchesFileLocation, std::vector<cv::Vec3d>& centers, std::vector<cv::Mat >& visibility, std::vector<std::vector<CCNF_patch_expert> >& patches, double& patchScaling);
    int GetViewIdx(const cv::Vec3d& rvec, int scale) const;
    int GetScaleIdx(const cv::Size size, const cv::Mat lmks) const;
    void Fit(const cv::Mat &im, const std::vector<int>& window_sizes);
    void Response(vector<cv::Mat>& patch_expert_responses, const PDM &pdm, cv::Mat& sim_img_to_ref, const cv::Mat_<float>& grayscale_image, int window_size, int scale);
    std::vector<int> Collect_visible_landmarks(vector<vector<cv::Mat> > visibilities, int scale, int view_id, int n);
    
    vector<vector<vector<Multi_SVR_PatchExpert>>> svr_templates; // scale: view : lmkIdx
    vector<vector<vector<CCNF_patch_expert>>> ccnf_templates; // scale: view : lmkIdx
    
    // The node connectivity for CCNF experts, at different window sizes and corresponding to separate edge features
    vector<vector<cv::Mat_<float> > > sigma_components;
    
    //Useful to pre-allocate data for im2col so that it is not allocated for every iteration and every patch
    vector< map<int, cv::Mat_<float> > > preallocated_im2col;
    
    vector<double> scales;
    vector<vector<cv::Vec3d>> views;
    vector<vector<cv::Mat>> visibilities;
    
private:
    void Response(const cv::Mat &roi, cv::Mat &response);
};


#endif /* PatchExperts_hpp */

