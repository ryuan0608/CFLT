//
//  CLMParameters.hpp
//  Pods
//
//  Created by RongYuan on 7/23/18.
//

#ifndef CLMParameters_hpp
#define CLMParameters_hpp

#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

struct CLMParameters
{
    
    // A number of RLMS or NU-RLMS iterations
    int num_optimisation_iteration;
    
    // Should pose be limited to 180 degrees frontal
    bool limit_pose;
    
    // Should face validation be done
    bool validate_detections;
    
    // Landmark detection validator boundary for correct detection, the regressor output 1 (perfect alignment) 0 (bad alignment),
    float validation_boundary;
    
    // Used when tracking is going well
    vector<int> window_sizes_small;
    
    // Used when initialising or tracking fails
    vector<int> window_sizes_init;
    
    // Used for the current frame
    vector<int> window_sizes_current;
    
    // How big is the tracking template that helps with large motions
    float face_template_scale;
    bool use_face_template;
    
    // Where to load the model from
    string model_location;
    
    // this is used for the smooting of response maps (KDE sigma)
    float sigma;
    
    float reg_factor;    // weight put to regularisation
    float weight_factor; // factor for weighted least squares
    
    // should multiple views be considered during reinit
    bool multi_view;
    
    // Should the results be visualised and reported to console
    bool quiet_mode;
    
    // Should the model be refined hierarchically (if available)
    bool refine_hierarchical;
    
    // Should the parameters be refined for different scales
    bool refine_parameters;
    
    CLMParameters();
    
private:
    void init();
};

#endif /* CLMParameters_hpp */
