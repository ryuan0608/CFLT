//
//  CLM.hpp
//
//  Constrained Local Model
//  (1) "Tadas Baltrusaitis, Peter Robinson, Louis-Philippe Morency,
//       Constrained Local Neural Fields for robust facial landmark detection in the wild, ICCV2013"
//
//  Created by Rong Yuan on 7/22/18.
//

#ifndef CLM_hpp
#define CLM_hpp

#include <stdio.h>
#include "PatchExperts.hpp"
#include "PDM.hpp"
#include "CLMParameters.hpp"

class CLM
{
public:
    // Constructor
    CLM();
    CLM(string filename);
    
    // Initialize rigid/non-rigid face parameters using initial landmark guesses
    void init(const vector<cv::Point> &lmks2d);
    void init(const vector<cv::Point2f> &lmks2d);
    void init(const cv::Mat &mLmks2d);
    
    // Reading the model in
    void Read(string name);
    
    // Do actual detection
    bool Localise(const cv::Mat &img, const CLMParameters &parameters, const vector<cv::Point> &init_lmks);
    
    // Fit face model (rigid & non-rigid parameters) to gray image
    bool Fit(const cv::Mat &img, const CLMParameters &parameters, const vector<cv::Point> &init_lmks);
    
    // Helper reading function
    bool Read_CLM(string clm_location);
    
    // Regularized Landmark Mean-shifts
    // Refer to (1) or CLM Documentation
    void RLMS(const vector<cv::Mat>& responses, const cv::Mat &baseShape2D, const cv::Mat& sim_img_to_ref,  bool rigid, int winSize, int viewId, int scale, const CLMParameters& parameters);
    // Mean-shifts
    // Refer to (1) or CLM Documentation
    void NonVectorisedMeanShift_precalc_kde(cv::Mat& out_mean_shifts, const vector<cv::Mat>& patch_expert_responses, const cv::Mat_<float> &dxs, const cv::Mat_<float> &dys, int resp_size, float a, int scale, int view_id, map<int, cv::Mat_<float> >& kde_resp_precalc);
    // RLMS implementation based on Ceres
    // Not used now, but expected to speed things up using ceres
    void Ceres_RLMS(const vector<cv::Mat>& responses, const cv::Mat &baseShape2D, const cv::Mat& sim_img_to_ref,  bool rigid, int winSize, int viewId, int scale, const CLMParameters& parameters);
    // Helper function to do NU_RLMS(Non-uniform RLMS)
    // Not used now, but expected to get rid of some outliers for robustness
    void GetWeightMatrix(cv::Mat_<float>& WeightMatrix, int scale, int view_id, const CLMParameters& parameters);
    
    PDM pdm;
    
    PatchExperts patchExperts;
    
    cv::Mat face_template;
    
    // A collection of hierarchical CLM models that can be used for refinement
    vector<CLM>                    hierarchical_models;
    vector<string>                    hierarchical_model_names;
    vector<vector<pair<int,int>>>    hierarchical_mapping;
    vector<CLMParameters>        hierarchical_params;
    
    // Indicator if eye model is there for eye detection
    bool                eye_model;
    
    
    bool loaded_successfully;
    //===========================================================================
    // Member variables that retain the state of the tracking (reflecting the state of the lastly tracked (detected) image
    
    // Lastly detect 2D model shape [x1,x2,...xn,y1,...yn]
    cv::Mat_<float>            detected_landmarks;
    
    map<int, cv::Mat_<float> > kde_resp_precalc;
    
    int non_rigid_optimisation_count = 0;
};

#endif /* CLM_hpp */
