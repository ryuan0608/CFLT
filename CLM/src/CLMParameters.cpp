//
//  CLMParameters.cpp
//  Pods
//
//  Created by RongYuan on 7/23/18.
//

#include "CLMParameters.hpp"

CLMParameters::CLMParameters()
{
    // initialise the default values
    init();
}

void CLMParameters::init()
{
    
    // number of iterations that will be performed at each scale
    num_optimisation_iteration = 3;
    
    // using an external face checker based on SVM
    validate_detections = true;
    
    // Using hierarchical refinement by default (can be turned off)
    refine_hierarchical = true;
    
    // Refining parameters by default
    refine_parameters = true;
    
    window_sizes_small = vector<int>(4);
    window_sizes_init = vector<int>(4);
    
    // For fast tracking
    window_sizes_small[0] = 0;
    window_sizes_small[1] = 9;
    window_sizes_small[2] = 7;
    window_sizes_small[3] = 0;
    
    // Just for initialisation
    window_sizes_init.at(0) = 11;
    window_sizes_init.at(1) = 9;
    window_sizes_init.at(2) = 7;
    window_sizes_init.at(3) = 5;
    
    face_template_scale = 0.3f;
    
    // For first frame use the initialisation
    window_sizes_current = window_sizes_init;
    
    
    sigma = 1.5f;
    reg_factor = 25.0f;
    weight_factor = 0.0f; // By default do not use NU-RLMS for videos as it does not work as well for them
    
    validation_boundary = 0.725f;
    
    limit_pose = true;
    multi_view = false;
    
    quiet_mode = false;
    
}

