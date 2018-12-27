
//  CLM.cpp
//
////  Constrained Local Model
//  (1) "Tadas Baltrusaitis, Peter Robinson, Louis-Philippe Morency,
//       Constrained Local Neural Fields for robust facial landmark detection in the wild, ICCV2013"
//
//  Created by Rong Yuan on 7/22/18.
//

#include "CLM.hpp"
#include "CLMConfig.h"

#include "LandmarkMapper.hpp"

// Constructor
CLM::CLM()
{
    Read(OPEN_FACE_DIR + "vip_106.txt");
}
CLM::CLM(string filename)
{
    Read(filename);
}
bool CLM::Read_CLM(string clm_location)
{
    // Location of modules
    ifstream locations(clm_location.c_str(), ios_base::in);
    if(!locations.is_open())
    {
        cout << "Couldn't open the CLM model file aborting" << endl;
        cout.flush();
        return false;
    }
    
    string line;
    
    vector<string> intensity_expert_locations;
    vector<string> ccnf_expert_locations;
    vector<string> cen_expert_locations;
    string early_term_loc;
    
    // The other module locations should be defined as relative paths from the main model
//    boost::filesystem::path root = boost::filesystem::path(clm_location).parent_path();
    
    // The main file contains the references to other files
    while (!locations.eof())
    {
        
        getline(locations, line);
        
        stringstream lineStream(line);
        
        string module;
        string location;
        
        // figure out which module is to be read from which file
        lineStream >> module;
        
        getline(lineStream, location);
        
        if(location.size() > 0)
            location.erase(location.begin()); // remove the first space
        
        // remove carriage return at the end for compatibility with unix systems
        if(location.size() > 0 && location.at(location.size()-1) == '\r')
        {
            location = location.substr(0, location.size()-1);
        }
        
        // append the lovstion to root location (boost syntax)
        string root = clm_location.substr(0, clm_location.find_last_of("/")+1);
        location = root + location;
        
        if (module.compare("PDM") == 0)
        {
            cout << "Reading the PDM module from: " << location << "....";
            bool read_success = pdm.Read(location);
            
            if (!read_success)
            {
                return false;
            }
            
            cout << "Done" << endl;
        }
        else if (module.compare("Triangulations") == 0)
        {
            cout << "Reading the Triangulations module from: " << location << "....";
            ifstream triangulationFile(location.c_str(), ios_base::in);
            
            if(!triangulationFile.is_open())
            {
                return false;
            }
            
            SkipComments(triangulationFile);
            
            int numViews;
            triangulationFile >> numViews;
            
            // read in the triangulations
            // not useful
            vector<cv::Mat> triangulations;
            triangulations.resize(numViews);
            
            for(int i = 0; i < numViews; ++i)
            {
                SkipComments(triangulationFile);
                ReadMat(triangulationFile, triangulations[i]);
            }
            cout << "Done" << endl;
        }
        else if(module.compare("PatchesIntensity") == 0)
        {
            intensity_expert_locations.push_back(location);
        }
        else if(module.compare("PatchesCCNF") == 0)
        {
            ccnf_expert_locations.push_back(location);
        }
    }
    
    // Initialise the patch experts
    bool read_success = patchExperts.Read(intensity_expert_locations, ccnf_expert_locations);
    
    return read_success;
}

void CLM::Read(string main_location)
{
    cout << "Reading the landmark detector/tracker from: " << main_location << endl;
    
    ifstream locations(main_location.c_str(), ios_base::in);
    if(!locations.is_open())
    {
        cout << "Couldn't open the model file, aborting" << endl;
        loaded_successfully = false;
        return;
    }
    string line;
    
    // Assume no eye model, unless read-in
    eye_model = false;
    
    // The main file contains the references to other files
    while (!locations.eof())
    {
        getline(locations, line);
        
        stringstream lineStream(line);
        
        string module;
        string location;
        
        // figure out which module is to be read from which file
        lineStream >> module;
        
        lineStream >> location;
        
        // remove carriage return at the end for compatibility with unix systems
        if(location.size() > 0 && location.at(location.size()-1) == '\r')
        {
            location = location.substr(0, location.size()-1);
        }
        
        
        // append to root
        string root = main_location.substr(0, main_location.find_last_of("/")+1);
        location = root + location;
        if (module.compare("LandmarkDetector") == 0)
        {
            cout << "Reading the landmark detector module from: " << location << endl;
            
            // The CLNF module includes the PDM and the patch experts
            bool read_success = Read_CLM(location);
            
            if(!read_success)
            {
                loaded_successfully = false;
                return;
            }
        }
        else if(module.compare("LandmarkDetector_part") == 0)
        {
            string part_name;
            lineStream >> part_name;
            cout << "Reading part based module...." << part_name << endl;
            
            vector<pair<int, int>> mappings;
            while(!lineStream.eof())
            {
                int ind_in_main;
                lineStream >> ind_in_main;
                
                int ind_in_part;
                lineStream >> ind_in_part;
                mappings.push_back(pair<int, int>(ind_in_main, ind_in_part));
            }
            
            this->hierarchical_mapping.push_back(mappings);
            
            CLM part_model(location);
            
            if (!part_model.loaded_successfully)
            {
                loaded_successfully = false;
                return;
            }
            
            this->hierarchical_models.push_back(part_model);
            
            this->hierarchical_model_names.push_back(part_name);
            
            // Making sure we look based on model directory
//            std::string root_loc = boost::filesystem::path(main_location).parent_path().string();
//            std::vector<string> sub_arguments{ OPEN_FACE_DIR };
            
            CLMParameters params;
            
            params.validate_detections = false;
            params.refine_hierarchical = false;
            params.refine_parameters = false;
            
            if(part_name.compare("left_eye") == 0 || part_name.compare("right_eye") == 0)
            {
                
                vector<int> windows_large;
                windows_large.push_back(5);
                windows_large.push_back(3);
                
                vector<int> windows_small;
                windows_small.push_back(5);
                windows_small.push_back(3);
                
                params.window_sizes_init = windows_large;
                params.window_sizes_small = windows_small;
                params.window_sizes_current = windows_large;
                
                params.reg_factor = 0.1;
                params.sigma = 2;
            }
            else if(part_name.compare("left_eye_28") == 0 || part_name.compare("right_eye_28") == 0)
            {
                vector<int> windows_large;
                windows_large.push_back(3);
                windows_large.push_back(5);
                windows_large.push_back(9);
                
                vector<int> windows_small;
                windows_small.push_back(3);
                windows_small.push_back(5);
                windows_small.push_back(9);
                
                params.window_sizes_init = windows_large;
                params.window_sizes_small = windows_small;
                params.window_sizes_current = windows_large;
                
                params.reg_factor = 0.5;
                params.sigma = 1.0;
                
                eye_model = true;
            }
            else if(part_name.compare("mouth") == 0)
            {
                vector<int> windows_large;
                windows_large.push_back(7);
                windows_large.push_back(7);
                
                vector<int> windows_small;
                windows_small.push_back(7);
                windows_small.push_back(7);
                
                params.window_sizes_init = windows_large;
                params.window_sizes_small = windows_small;
                params.window_sizes_current = windows_large;
                
                params.reg_factor = 1.0;
                params.sigma = 2.0;
            }
            else if(part_name.compare("brow") == 0)
            {
                vector<int> windows_large;
                windows_large.push_back(11);
                windows_large.push_back(9);
                
                vector<int> windows_small;
                windows_small.push_back(11);
                windows_small.push_back(9);
                
                params.window_sizes_init = windows_large;
                params.window_sizes_small = windows_small;
                params.window_sizes_current = windows_large;
                
                params.reg_factor = 10.0;
                params.sigma = 3.5;
            }
            else if(part_name.compare("inner") == 0)
            {
                vector<int> windows_large;
                windows_large.push_back(9);
                
                vector<int> windows_small;
                windows_small.push_back(9);
                
                params.window_sizes_init = windows_large;
                params.window_sizes_small = windows_small;
                params.window_sizes_current = windows_large;
                
                params.reg_factor = 2.5;
                params.sigma = 1.75;
                params.weight_factor = 2.5;
            }
            
            this->hierarchical_params.push_back(params);
            
            cout << "Done" << endl;
        }
    }
    
    detected_landmarks.create(2 * pdm.nPts(), 1);
    detected_landmarks.setTo(0);
    
    
    // Initialising default values for the rest of the variables
    
    loaded_successfully = true;
}

// Solve PnP based on provided landmark estimates
// NOTE THAT lmks2d doesn't have to be 68 landmarks.
// Now the mapper supports conversion from 106 to 76/68/5 landmarks,
// and it's easily extendible. Read the code[mapFrom106Lmks] for detail
void CLM::init(const vector<cv::Point2f> &lmks2d)
{
    // reshape to 68 landmarks first
    vector<cv::Point2f> lmks68;
    if(lmks2d.size() == 68)
        lmks68 = lmks2d;
    else
        lmks68 = mapFrom106Lmks(lmks2d, 68);
    
    // 0. PnP gives initial pose info
    cv::Mat shape;
    pdm.getShape(shape);
    
    const int n = int(lmks68.size());
    
    cv::Mat shapeSample = mapFrom106Lmks(shape.t(), n);
    shapeSample = shapeSample.rowRange(18, 68);
    
    vector<cv::Point2f> lmks2dSample;
    for(int i = 18; i < 68; i++)
        lmks2dSample.push_back(lmks68[i]);
    
    cv::Vec3d tmpRvec = pdm.getRvec(), tmpTvec = pdm.getTvec();
    
    if ((tmpTvec == cv::Vec3d(0, 0, 0) && tmpRvec == cv::Vec3d(0, 0, 0)) || (tmpTvec[2] < 0)){
        cv::solvePnP(shapeSample, lmks2dSample, intrinsic(), cv::Mat(), tmpRvec, tmpTvec, false);
    }
    else{
        cv::solvePnP(shapeSample, lmks2dSample, intrinsic(), cv::Mat(), tmpRvec, tmpTvec, true);
    }
    pdm.setRvec(tmpRvec);
    pdm.setTvec(tmpTvec);
//    int k = pdm.nDim();
//    // 1. compute jacobian
//    for(int iter = 0; iter < 3; iter++)
//    {
//        cv::Mat J, J68(68*2, 6+k, CV_32F);
//        pdm.jacobian(J);
//        const vector<vector<int>>* ptr = &LMK106_TO_LMK68;
//        for(int i = 0; i < 68; i++)
//        {
//            if(ptr->at(i).size() == 1)
//            {
//                int idx = ptr->at(i)[0] - 1;
//                J.row(2*idx).copyTo(J68.row(2*i));
//                J.row(2*idx+1).copyTo(J68.row(2*i+1));
//            }
//            else
//            {
//                int idx1 = ptr->at(i)[0] - 1, idx2 = ptr->at(i)[1] - 1;
//                J68.row(2*i) = (J.row(2*idx1) + J.row(2*idx2)) / 2;
//                J68.row(2*i+1) = (J.row(2*idx1+1) + J.row(2*idx2+1)) / 2;
//            }
//        }
//        J68.copyTo(J);
//        // 2. compute residual
//        cv::Mat v(68*2, 1, CV_32F);
//        cv::Vec3d rvec, tvec;
//        rvec = pdm.getRvec();
//        tvec = pdm.getTvec();
//        pdm.getShape(shape);
//        cv::Mat projected106(106, 2, CV_32F), projected68;
//        vector<cv::Point2f> projected;
//        cv::projectPoints(shape.t(), rvec, tvec, intrinsic(), cv::Mat(), projected);
//        for(int i = 0; i < 106; i++)
//        {
//            projected106.at<float>(i, 0) = projected[i].x;
//            projected106.at<float>(i, 1) = projected[i].y;
//        }
//        projected68 = mapFrom106Lmks(projected106, 68);
//        for(int i = 0; i < 68; i++)
//        {
//            v.at<float>(2*i) = projected68.at<float>(i, 0) - lmks2d[i].x;
//            v.at<float>(2*i+1) = projected68.at<float>(i, 1) - lmks2d[i].y;
//        }
//        // 3. compute regularization term
//        //    setting the regularisation to the inverse of eigenvalues
//
//        const float r = 25.0f;
//        cv::Mat regMatrix = cv::Mat::zeros(6+k, 6+k, CV_32F);
//        cv::Mat evs = pdm.evs();
//        for(int i = 6; i < 6 + k; i++)
//            regMatrix.at<float>(i, i) = r / evs.at<float>(i-6);
//        // 4. do actual optimization
//        cv::Mat term1 = J.t() * J + regMatrix;
//        cv::Mat term2 = J.t() * v;
//        cv::Mat currParams;
//        pdm.getParams(currParams);
//        term2(cv::Rect(0, 6, 1, k)) = term2(cv::Rect(0, 6, 1, k)) - regMatrix(cv::Rect(6, 6, k, k)) * currParams;
//        cv::Mat delta;
//        cv::solve(term1, term2, delta, CV_CHOLESKY);
//        delta *= -0.5;
//        // 5. update
//        rvec += cv::Vec3d(delta.rowRange(0, 3));
//        tvec += cv::Vec3d(delta.rowRange(3, 6));
//        pdm.setRvec(rvec);
//        pdm.setTvec(tvec);
//        currParams += delta.rowRange(6, 6+k);
//        int count = 0;
    //    for(int m = 0; m < k; m++)
    //    {
    //        float ev = evs.at<float>(m);
    //        if(currParams.at<float>(m) > 3 * sqrt(ev))
    //        {
    //            currParams.at<float>(m) = 3 * sqrt(ev);
    //            ++count;
    //        }
    //        else if(currParams.at<float>(m) < -3 * sqrt(ev))
    //        {
    //            currParams.at<float>(k) = -3 * sqrt(ev);
    //            ++count;
    //        }
    //        cout << currParams.at<float>(m) << "vs" << 3*sqrt(ev) << endl;
    //    }
//        pdm.setParams(currParams);
//    }
}
void CLM::init(const vector<cv::Point> &iLmks2d)
{
    vector<cv::Point2f> lmks2d;
    for(int i = 0; i < iLmks2d.size(); i++)
        lmks2d.push_back(cv::Point2f(iLmks2d[i]));
    init(lmks2d);
}
void CLM::init(const cv::Mat &mLmks2d)
{
    vector<cv::Point2f> lmks2d;
    for(int i = 0; i < mLmks2d.rows; i++)
        lmks2d.push_back(cv::Point2f(mLmks2d.row(i)));
    init(lmks2d);
}

bool CLM::Localise(const cv::Mat &img, const CLMParameters &params, const vector<cv::Point> &init_lmks)
{
    // Fits from the current estimate of local and global parameters in the model
    bool fit_success = Fit(img, params, init_lmks);
    
    // Store the landmarks converged on in detected_landmarks
    detected_landmarks = pdm.calc2DShape();
    
    /* NOT USING HIERARCHICAL PART MODEL NOW
    // This requires local part models trained separately
    // and is believed to improve landmark accuracy under large scale expressions
    if(params.refine_hierarchical && hierarchical_models.size() > 0)
    {
        // Do the hierarchical models in parallel
        for(int part_model = 0; part_model < hierarchical_models.size(); part_model++)
        {
            int n_part_points = hierarchical_models[part_model].pdm.nPts();

            vector<pair<int, int>> mappings = this->hierarchical_mapping[part_model];

            cv::Mat part_model_locs = cv::Mat::zeros(n_part_points, 2, CV_32F);

            // Extract the corresponding landmarks
            for (size_t mapping_ind = 0; mapping_ind < mappings.size(); ++mapping_ind)
                detected_landmarks.row(mappings[mapping_ind].first).copyTo(part_model_locs.row(mappings[mapping_ind].second));

            // Fit the part based model PDM
            hierarchical_models[part_model].pdm.calcParams(part_model_locs);
            
            this->hierarchical_params[part_model].window_sizes_current = this->hierarchical_params[part_model].window_sizes_init;
            
            // Do the actual landmark detection
            hierarchical_models[part_model].Localise(img, hierarchical_params[part_model]);
        }
        
        // Recompute main model based on the fit part models
        for (size_t part_model = 0; part_model < hierarchical_models.size(); ++part_model)
        {
            vector<pair<int, int>> mappings = this->hierarchical_mapping[part_model];
            
            // Reincorporate the models into main tracker
            for (size_t mapping_ind = 0; mapping_ind < mappings.size(); ++mapping_ind)
            {
                hierarchical_models[part_model].detected_landmarks.row(mappings[mapping_ind].second).copyTo(detected_landmarks.row(mappings[mapping_ind].first));
            }
        }
        
        //            pdm.calcParams(detected_landmarks);
        //            detected_landmarks = pdm.calc2DShape();
        
    }
     */
    return fit_success;
}

// Fit face model (rigid & non-rigid parameters) to gray image
bool CLM::Fit(const cv::Mat &img, const CLMParameters &parameters, const vector<cv::Point> &init_lmks)
{
    
    cv::Mat params;
    pdm.getParams(params);
    
    const int n = pdm.nPts();
    // placeholders
    vector<cv::Mat> responses(n);
    cv::Mat img2refTransform;
    
    // start with model parameters
    // and subject to changes
    CLMParameters tmpParameters = parameters;
    
    bool optimise_non_rigid = (non_rigid_optimisation_count % 5 == 0);
    // Optimise the model across a number of areas of interest (usually in descending window size and ascending scale size)
    cv::Mat intr = intrinsic();
    cv::Mat initialShape2D = pdm.calc2DShape();

    // update initial 2d shape with initial landmark locations
    // note that a mapping is required if the initial landmarks is not equal in number to shape
    for(int i = 48; i < 68; i++)
    {
        if(LMK106_TO_LMK68[i].size() > 1)
            continue;
        // only update those matched landmarks
        int index = LMK106_TO_LMK68[i][0] - 1;
        initialShape2D.at<float>(index, 0) = init_lmks[i].x;
        initialShape2D.at<float>(index, 1) = init_lmks[i].y;
    }
    
    
    int scale = patchExperts.GetScaleIdx(img.size(), initialShape2D);
    int winSize = parameters.window_sizes_current[scale];

    // The patch expert response computation
    patchExperts.Response(responses, pdm, img2refTransform, img, winSize, scale);
    if(parameters.refine_parameters == true)
    {
        int scale_max = scale >= 2 ? 2 : scale;

        // Adapt the parameters based on scale (want to reduce regularisation as scale increases, but increase sigma and Tikhonov)
        tmpParameters.reg_factor = parameters.reg_factor - 15 * log(patchExperts.scales[scale_max]/0.25)/log(2);

        if(tmpParameters.reg_factor <= 0)
            tmpParameters.reg_factor = 0.001;

        tmpParameters.sigma = parameters.sigma + 0.25 * log(patchExperts.scales[scale_max]/0.25)/log(2);
        tmpParameters.weight_factor = parameters.weight_factor + 2 * parameters.weight_factor *  log(patchExperts.scales[scale_max]/0.25)/log(2);
    }

    const int viewId = patchExperts.GetViewIdx(pdm.getRvec(), scale);
    // rigid optimisation
    RLMS(responses, initialShape2D, img2refTransform, true, winSize, viewId, scale, tmpParameters);
    // non-rigid optimisation
    if(optimise_non_rigid)
        RLMS(responses, initialShape2D, img2refTransform, false, winSize, viewId, scale, tmpParameters);

    if(optimise_non_rigid)
        non_rigid_optimisation_count = 0;
    else
        ++non_rigid_optimisation_count;
    return true;
}

void CLM::RLMS(const vector<cv::Mat>& responses, const cv::Mat &baseShape2D, const cv::Mat& img2refTrans,  bool rigid, int winSize, int viewId, int scale, const CLMParameters& parameters)
{
    const int n = pdm.nPts();
    const int m = pdm.nDim();
    
    cv::Mat currShape2D, prevShape2D;
    
    // Pre-calculate the regularisation term
    cv::Mat regMatrix;
    const cv::Mat evs = pdm.evs();
    if(rigid)
    {
        regMatrix = cv::Mat::zeros(6, 6, CV_32F);
    }
    else
    {
        regMatrix = cv::Mat::zeros(6+m, 6+m, CV_32F);
        // Setting the regularisation to the inverse of eigenvalues
        const float r = parameters.reg_factor;
        
        for(int i = 6; i < 6 + m; i++)
            regMatrix.at<float>(i, i) = r / evs.at<float>(i-6);
    }
    
    cv::Mat_<float> dxs, dys;
    
    // The preallocated memory for the mean shifts
    cv::Mat meanShifts(2 * n, 1, CV_32F);
    
    // start optimization process
    cv::Mat J;
    const float convergenceThreshold = 0.01;
    cv::Mat ref2imgTrans = img2refTrans.inv(cv::DECOMP_LU);
    for(int i = 0; i < parameters.num_optimisation_iteration; i++)
    {
        currShape2D = pdm.calc2DShape();
        
        if(i > 0)
        {
            if(cv::norm(currShape2D - prevShape2D) < convergenceThreshold)
                break;
        }
        currShape2D.copyTo(prevShape2D);
        // initialize Jacobian to be zero matrix
        if(rigid)
            pdm.jacobianRT(J);
        else
            pdm.jacobian(J);
        
        // useful for mean shift calculation
        const float sigma = parameters.sigma;
        float a = -0.5/(sigma * sigma);
        
        
        cv::Mat_<float> offsets;
        cv::Mat((currShape2D - baseShape2D) * cv::Mat(img2refTrans).t()).convertTo(offsets, CV_32F);
        
        dxs = offsets.col(0) + (winSize-1)/2;
        dys = offsets.col(1) + (winSize-1)/2;
        
        NonVectorisedMeanShift_precalc_kde(meanShifts, responses, dxs, dys, winSize, a, scale, viewId, kde_resp_precalc);
        
        // Now transform the mean shifts to the the image reference frame, as opposed to one of ref shape (object space)
        cv::Mat_<float> mean_shifts_2D = (meanShifts.reshape(1, 2)).t();
        
        mean_shifts_2D = mean_shifts_2D * cv::Mat(ref2imgTrans).t();
        meanShifts = cv::Mat(mean_shifts_2D.t()).reshape(1, n*2);
        
//        // remove non-visible observations
//        for(int i = 0; i < n; ++i)
//        {
//            // if patch unavailable for current index
//            if(patchExperts.visibilities[scale][viewId].at<int>(i, 0) == 0)
//            {
//                J.rowRange(2*i, 2*i+2).setTo(0);
//                meanShifts.rowRange(2*i, 2*i+2).setTo(0);
//            }
//        }
        
        
//        cv::Mat_<float> W;
//        GetWeightMatrix(W, scale, viewId, parameters);
        
        
//        cv::Mat J_w_t = J.t();
//        cv::Mat J_w_t_m = J_w_t * meanShifts;
//        cv::Mat currParams = pdm.getParams();
//        if(!rigid)
//            J_w_t_m(cv::Rect(0,6,1, m)) = J_w_t_m(cv::Rect(0,6,1, m)) - regMatrix(cv::Rect(6,6, m, m)) * currParams;
//
//        cv::Mat Hessian = regMatrix.clone();
//        // Perform matrix multiplication in OpenBLAS (fortran call)
//        float alpha1 = 1.0;
//        float beta1 = 1.0;
//        char N[2]; N[0] = 'N';
//        sgemm_(N, N, &J.cols, &J_w_t.rows, &J_w_t.cols, &alpha1, (float*)J.data, &J.cols, (float*)J_w_t.data, &J_w_t.cols, &beta1, (float*)Hessian.data, &J.cols);
//        // Solve for the parameter update (from Baltrusaitis 2013 based on eq (36) Saragih 2011)
//        cv::Mat delta;
//        cv::solve(Hessian, J_w_t_m, delta, CV_CHOLESKY);
        
        // Compute parameter update
        cv::Mat term1 = J.t() * J + regMatrix;
        cv::Mat term2 = J.t() * meanShifts;
        cv::Mat currParams;
        pdm.getParams(currParams);
        if(!rigid)
            term2(cv::Rect(0, 6, 1, m)) = term2(cv::Rect(0, 6, 1, m)) - regMatrix(cv::Rect(6, 6, m, m)) * currParams;
        // Solve for the parameter update
        /* from Baltrusaitis 2013 based on eq (36) Saragih 2011 */
        cv::Mat delta;
        cv::solve(term1, term2, delta, CV_CHOLESKY);
        
        // update parameters
        cv::Vec3d rvec = pdm.getRvec(), tvec = pdm.getTvec();
        rvec += cv::Vec3d(delta.rowRange(0, 3));
        tvec += cv::Vec3d(delta.rowRange(3, 6));
        pdm.setRvec(rvec);
        pdm.setTvec(tvec);
        
        if(!rigid)
        {
            currParams += delta.rowRange(6, 6+m);
            int count = 0;
            for(int k = 0; k < m; k++)
            {
                float ev = evs.at<float>(k);
                if(currParams.at<float>(k) > 3 * sqrt(ev))
                {
                    currParams.at<float>(k) = 3 * sqrt(ev);
                    ++count;
                }
                else if(currParams.at<float>(k) < -3 * sqrt(ev))
                {
                    currParams.at<float>(k) = -3 * sqrt(ev);
                    ++count;
                }
            }
            pdm.setParams(currParams);
        }
        
    }
}

void CLM::Ceres_RLMS(const vector<cv::Mat>& responses, const cv::Mat &baseShape2D, const cv::Mat& img2refTrans,  bool rigid, int winSize, int viewId, int scale, const CLMParameters& parameters)
{
    const int n = pdm.nPts();
    const int m = pdm.nDim();
    
    cv::Mat currShape2D, prevShape2D;
    
    // Pre-calculate the regularisation term
    cv::Mat regMatrix;
    const cv::Mat evs = pdm.evs();
    if(rigid)
    {
        regMatrix = cv::Mat::zeros(6, 6, CV_32F);
    }
    else
    {
        regMatrix = cv::Mat::zeros(6+m, 6+m, CV_32F);
        // Setting the regularisation to the inverse of eigenvalues
        const float r = parameters.reg_factor;
        
        for(int i = 6; i < 6 + m; i++)
            regMatrix.at<float>(i, i) = r / evs.at<float>(i-6);
    }
    
    cv::Mat_<float> dxs, dys;
    
    // The preallocated memory for the mean shifts
    cv::Mat meanShifts(2 * n, 1, CV_32F);
    
    // start optimization process
    cv::Mat J;
    const float convergenceThreshold = 0.01;
    cv::Mat ref2imgTrans = img2refTrans.inv(cv::DECOMP_LU);
    for(int i = 0; i < parameters.num_optimisation_iteration; i++)
    {
        currShape2D = pdm.calc2DShape();
        
        if(i > 0)
        {
            if(cv::norm(currShape2D - prevShape2D) < convergenceThreshold)
                break;
        }
        currShape2D.copyTo(prevShape2D);
        // initialize Jacobian to be zero matrix
        if(rigid)
            pdm.jacobianRT(J);
        else
            pdm.jacobian(J);
        
        // useful for mean shift calculation
        const float sigma = parameters.sigma;
        float a = -0.5/(sigma * sigma);
        
        
        cv::Mat_<float> offsets;
        cv::Mat((currShape2D - baseShape2D) * cv::Mat(img2refTrans).t()).convertTo(offsets, CV_32F);
        
        dxs = offsets.col(0) + (winSize-1)/2;
        dys = offsets.col(1) + (winSize-1)/2;
        
        NonVectorisedMeanShift_precalc_kde(meanShifts, responses, dxs, dys, winSize, a, scale, viewId, kde_resp_precalc);
        
        // Now transform the mean shifts to the the image reference frame, as opposed to one of ref shape (object space)
        cv::Mat_<float> mean_shifts_2D = (meanShifts.reshape(1, 2)).t();
        
        mean_shifts_2D = mean_shifts_2D * cv::Mat(ref2imgTrans).t();
        meanShifts = cv::Mat(mean_shifts_2D.t()).reshape(1, n*2);
        
        //        // remove non-visible observations
        //        for(int i = 0; i < n; ++i)
        //        {
        //            // if patch unavailable for current index
        //            if(patchExperts.visibilities[scale][viewId].at<int>(i, 0) == 0)
        //            {
        //                J.rowRange(2*i, 2*i+2).setTo(0);
        //                meanShifts.rowRange(2*i, 2*i+2).setTo(0);
        //            }
        //        }
        
        
        //        cv::Mat_<float> W;
        //        GetWeightMatrix(W, scale, viewId, parameters);
        
        
        //        cv::Mat J_w_t = J.t();
        //        cv::Mat J_w_t_m = J_w_t * meanShifts;
        //        cv::Mat currParams = pdm.getParams();
        //        if(!rigid)
        //            J_w_t_m(cv::Rect(0,6,1, m)) = J_w_t_m(cv::Rect(0,6,1, m)) - regMatrix(cv::Rect(6,6, m, m)) * currParams;
        //
        //        cv::Mat Hessian = regMatrix.clone();
        //        // Perform matrix multiplication in OpenBLAS (fortran call)
        //        float alpha1 = 1.0;
        //        float beta1 = 1.0;
        //        char N[2]; N[0] = 'N';
        //        sgemm_(N, N, &J.cols, &J_w_t.rows, &J_w_t.cols, &alpha1, (float*)J.data, &J.cols, (float*)J_w_t.data, &J_w_t.cols, &beta1, (float*)Hessian.data, &J.cols);
        //        // Solve for the parameter update (from Baltrusaitis 2013 based on eq (36) Saragih 2011)
        //        cv::Mat delta;
        //        cv::solve(Hessian, J_w_t_m, delta, CV_CHOLESKY);
        
        cv::Mat term1 = J.t() * J + regMatrix;
        cv::Mat term2 = J.t() * meanShifts;
        cv::Mat currParams;
        pdm.getParams(currParams);
        if(!rigid)
            term2(cv::Rect(0, 6, 1, m)) = term2(cv::Rect(0, 6, 1, m)) - regMatrix(cv::Rect(6, 6, m, m)) * currParams;
        // Solve for the parameter update (from Baltrusaitis 2013 based on eq (36) Saragih 2011)
        cv::Mat delta;
        cv::solve(term1, term2, delta, CV_CHOLESKY);
        
        // update parameters
        cv::Vec3d rvec = pdm.getRvec(), tvec = pdm.getTvec();
        rvec += cv::Vec3d(delta.rowRange(0, 3));
        tvec += cv::Vec3d(delta.rowRange(3, 6));
        pdm.setRvec(rvec);
        pdm.setTvec(tvec);
        
        if(!rigid)
        {
            currParams += delta.rowRange(6, 6+m);
            for(int k = 0; k < m; k++)
            {
                float ev = evs.at<float>(k);
//                if(currParams.at<float>(k) > 3 * sqrt(ev))
//                    currParams.at<float>(k) = 3 * sqrt(ev);
//                else if(currParams.at<float>(k) < -3 * sqrt(ev))
//                    currParams.at<float>(k) = -3 * sqrt(ev);
            }
            pdm.setParams(currParams);
        }
    }
}

void CLM::NonVectorisedMeanShift_precalc_kde(cv::Mat& out_mean_shifts, const vector<cv::Mat>& patch_expert_responses, const cv::Mat_<float> &dxs, const cv::Mat_<float> &dys, int resp_size, float a, int scale, int view_id, map<int, cv::Mat_<float> >& kde_resp_precalc)
{
    
    int n = dxs.rows;
    
    cv::Mat_<float> kde_resp;
    float step_size = 0.1;
    
    // if this has not been precomputed, precompute it, otherwise use it
    if(kde_resp_precalc.find(resp_size) == kde_resp_precalc.end())
    {
        kde_resp = cv::Mat_<float>((int)((resp_size / step_size)*(resp_size/step_size)), resp_size * resp_size);
        cv::MatIterator_<float> kde_it = kde_resp.begin();
        
        for(int x = 0; x < resp_size/step_size; x++)
        {
            float dx = x * step_size;
            for(int y = 0; y < resp_size/step_size; y++)
            {
                float dy = y * step_size;
                
                int ii,jj;
                float v,vx,vy;
                
                for(ii = 0; ii < resp_size; ii++)
                {
                    vx = (dy-ii)*(dy-ii);
                    for(jj = 0; jj < resp_size; jj++)
                    {
                        vy = (dx-jj)*(dx-jj);
                        
                        // the KDE evaluation of that point
                        v = exp(a*(vx+vy));
                        
                        *kde_it++ = v;
                    }
                }
            }
        }
        
        kde_resp_precalc[resp_size] = kde_resp.clone();
    }
    else
    {
        // use the precomputed version
        kde_resp = kde_resp_precalc.find(resp_size)->second;
    }
    
    // for every point (patch) calculating mean-shift
    for(int i = 0; i < n; i++)
    {
        // indices of dx, dy
        float dx = dxs.at<float>(i);
        float dy = dys.at<float>(i);
        
        // Ensure that we are within bounds (important for precalculation)
        if(dx < 0)
            dx = 0;
        if(dy < 0)
            dy = 0;
        if(dx > resp_size - step_size)
            dx = resp_size - step_size;
        if(dy > resp_size - step_size)
            dy = resp_size - step_size;
        
        // Pick the row from precalculated kde that approximates the current dx, dy best
        int closest_col = (int)(dy /step_size + 0.5); // Plus 0.5 is there, as C++ rounds down with int cast
        int closest_row = (int)(dx /step_size + 0.5); // Plus 0.5 is there, as C++ rounds down with int cast
        
        int idx = closest_row * ((int)(resp_size/step_size + 0.5)) + closest_col; // Plus 0.5 is there, as C++ rounds down with int cast
        
        cv::MatIterator_<float> kde_it = kde_resp.begin() + kde_resp.cols*idx;
        
        float mx=0.0;
        float my=0.0;
        float sum=0.0;
        
        // Iterate over the patch responses here
        cv::MatConstIterator_<float> p = patch_expert_responses[i].begin<float>();
        
        // TODO maybe do through MatMuls instead?
        for(int ii = 0; ii < resp_size; ii++)
        {
            for(int jj = 0; jj < resp_size; jj++)
            {
                
                // the KDE evaluation of that point multiplied by the probability at the current, xi, yi
                float v = (*p++) * (*kde_it++);
                
                sum += v;
                
                // mean shift in x and y
                mx += v*jj;
                my += v*ii;
                
            }
        }
        
        float msx = (mx/sum - dx);
        float msy = (my/sum - dy);
        
        out_mean_shifts.at<float>(2*i,0) = msx;
        out_mean_shifts.at<float>(2*i+1,0) = msy;
    }
    
}

void CLM::GetWeightMatrix(cv::Mat_<float>& WeightMatrix, int scale, int view_id, const CLMParameters& parameters)
{
    int n = pdm.nPts();
    
    // Is the weight matrix needed at all
    if(parameters.weight_factor > 0)
    {
        WeightMatrix = cv::Mat_<float>::zeros(n*2, n*2);
        
        for (int p=0; p < n; p++)
        {
            if(!patchExperts.ccnf_templates.empty())
            {
                
                // for the x dimension
                WeightMatrix.at<float>(p,p) = WeightMatrix.at<float>(p,p)  + patchExperts.ccnf_templates[scale][view_id][p].patch_confidence;
                
                // for they y dimension
                WeightMatrix.at<float>(p+n,p+n) = WeightMatrix.at<float>(p,p);
                
            }
            else
            {
                // Across the modalities add the confidences
                for(size_t pc=0; pc < patchExperts.svr_templates[scale][view_id][p].svr_patch_experts.size(); pc++)
                {
                    // for the x dimension
                    WeightMatrix.at<float>(p,p) = WeightMatrix.at<float>(p,p)  + patchExperts.svr_templates[scale][view_id][p].svr_patch_experts.at(pc).confidence;
                }
                // for the y dimension
                WeightMatrix.at<float>(p+n,p+n) = WeightMatrix.at<float>(p,p);
            }
        }
        WeightMatrix = parameters.weight_factor * WeightMatrix;
    }
    else
    {
        WeightMatrix = cv::Mat_<float>::eye(n*2, n*2);
    }
    
}
