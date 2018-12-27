//
//  PatchExperts.cpp
//  Pods
//
//  Created by RongYuan on 7/20/18.
//

#include <stdio.h>
#include "PatchExperts.hpp"
#include "CLM_utils.hpp"

//===========================================================================
bool PatchExperts::Read(vector<string> intensity_svr_expert_locations, vector<string> intensity_ccnf_expert_locations)
{
    
    // initialise the SVR intensity patch expert parameters
    int num_intensity_svr = int(intensity_svr_expert_locations.size());
    
    views.resize(num_intensity_svr);
    visibilities.resize(num_intensity_svr);
    scales.resize(num_intensity_svr);
    
    svr_templates.resize(num_intensity_svr);
    
    // Reading in SVR intensity patch experts for each scales it is defined in
    for(int scale = 0; scale < num_intensity_svr; ++scale)
    {
        string location = intensity_svr_expert_locations[scale];
        cout << "Reading the intensity SVR patch experts from: " << location << "....";
        bool success_read = Read_SVR_patch_experts(location,  views[scale], visibilities[scale], svr_templates[scale], scales[scale]);
        if (!success_read)
        {
            return false;
        }
    }
    
    // Initialise and read CCNF patch experts (currently only intensity based),
    int num_intensity_ccnf = intensity_ccnf_expert_locations.size();
    
    // CCNF experts override the SVR ones
    if(num_intensity_ccnf > 0)
    {
        views.resize(num_intensity_ccnf);
        visibilities.resize(num_intensity_ccnf);
        scales.resize(num_intensity_ccnf);
        ccnf_templates.resize(num_intensity_ccnf);
    }
    for(int scale = 0; scale < num_intensity_ccnf; ++scale)
    {
        string location = intensity_ccnf_expert_locations[scale];
        cout << "Reading the intensity CCNF patch experts from: " << location << "....";
        bool success_read = Read_CCNF_patch_experts(location, views[scale], visibilities[scale], ccnf_templates[scale], scales[scale]);
        
        if (!success_read)
        {
            return false;
        }
        
        if (scale == 0)
        {
            preallocated_im2col.resize(ccnf_templates[0][0].size());
        }
    }
    return true;
}

void PatchExperts::Response(vector<cv::Mat>& patch_expert_responses, const PDM &pdm, cv::Mat& sim_img_to_ref, const cv::Mat_<float>& grayscale_image, int window_size, int scale)
{
    cv::Vec3d rvec = pdm.getRvec(), tvec = pdm.getTvec();
    int view_id = GetViewIdx(rvec, scale);
    
    int n = pdm.nPts();
    
    // Compute the reference shape
    vector<cv::Point3f> lmks3d;
    cv::Mat shape;
    pdm.getShape(shape);
    shape = shape.t();
    vector<cv::Point2f> lmks2d;
    for(int i = 0; i < n; i++)
        lmks3d.push_back(cv::Point3f(shape.row(i)));
    cv::Mat reference_shape_2D(n, 2, CV_32F), image_shape(n, 2, CV_32F);
    cv::projectPoints(lmks3d, rvec, tvec, intrinsic(), cv::Mat(), lmks2d);
    for(int i = 0; i < n; i++)
    {
        image_shape.at<float>(i, 0) = lmks2d[i].x;
        image_shape.at<float>(i, 1) = lmks2d[i].y;
    }
    reference_shape_2D = shape.colRange(0, 2).clone();
    reference_shape_2D *= scales[scale];
    
    sim_img_to_ref = AlignShapesWithScale_f(image_shape, reference_shape_2D);
    cv::Mat sim_ref_to_img = sim_img_to_ref.inv(cv::DECOMP_LU);
    
    float a1 = sim_ref_to_img.at<float>(0, 0);
    float b1 = -sim_ref_to_img.at<float>(0, 1);
    
    bool use_ccnf = !this->ccnf_templates.empty();
    
    
    // If using CCNF patch experts might need to precalculate Sigmas
    if (use_ccnf)
    {
        vector<cv::Mat_<float> > sigma_components;
        
        // Retrieve the correct sigma component size
        for (size_t w_size = 0; w_size < this->sigma_components.size(); ++w_size)
        {
            if (!this->sigma_components[w_size].empty())
            {
                if (window_size*window_size == this->sigma_components[w_size][0].rows)
                {
                    sigma_components = this->sigma_components[w_size];
                }
            }
        }
        
        // Go through all of the landmarks and compute the Sigma for each
        for (int lmark = 0; lmark < n; lmark++)
        {
            // Only for visible landmarks
            if (visibilities[scale][view_id].at<int>(lmark, 0))
            {
                // Precompute sigmas if they are not computed yet
                ccnf_templates[scale][view_id][lmark].ComputeSigmas(sigma_components, window_size);
            }
        }
    }
    
    
    // We do not want to create threads for invisible landmarks, so construct an index of visible ones
    std::vector<int> vis_lmk = Collect_visible_landmarks(visibilities, scale, view_id, n);
    
    for (int i = 0; i < n; i++)
    {
        int ind = i;
        
        int area_of_interest_width, area_of_interest_height;
        if (use_ccnf)
        {
            area_of_interest_width = window_size + ccnf_templates[scale][view_id][ind].width - 1;
            area_of_interest_height = window_size + ccnf_templates[scale][view_id][ind].height - 1;
        }
        else
        {
            area_of_interest_width = window_size + svr_templates[scale][view_id][ind].width - 1;
            area_of_interest_height = window_size + svr_templates[scale][view_id][ind].height - 1;
        }
        
        
        // scale and rotate to mean shape to reference frame
        cv::Mat sim = (cv::Mat_<float>(2, 3) << a1, -b1, image_shape.at<float>(ind, 0) - a1 * (area_of_interest_width - 1.0f) / 2.0f + b1 * (area_of_interest_width - 1.0f) / 2.0f, b1, a1, image_shape.at<float>(ind, 1) - a1 * (area_of_interest_width - 1.0f) / 2.0f - b1 * (area_of_interest_width - 1.0f) / 2.0f);
        
        // Extract the region of interest around the current landmark location
        cv::Mat_<float> area_of_interest(area_of_interest_height, area_of_interest_width, CV_32F);
        
        cv::warpAffine(grayscale_image, area_of_interest, sim, area_of_interest.size(), cv::WARP_INVERSE_MAP + CV_INTER_LINEAR);
        
        
        if (!ccnf_templates.empty())
        {
            // get the correct size response window
            patch_expert_responses[ind] = cv::Mat_<float>(window_size, window_size);
            
            int im2col_size = area_of_interest_width * area_of_interest_height;
            
            cv::Mat_<float> prealloc_mat = preallocated_im2col[ind][im2col_size];
            
//            ccnf_templates[scale][view_id][ind].ResponseOpenBlas(area_of_interest, patch_expert_responses[ind], prealloc_mat);
            
//            preallocated_im2col[ind][im2col_size] = prealloc_mat;
            
            // Below is an alternative way to compute the same, but that uses FFT instead of OpenBLAS
             ccnf_templates[scale][view_id][ind].Response(area_of_interest, patch_expert_responses[ind]);
            
        }
        else
        {
            // get the correct size response window
            patch_expert_responses[ind] = cv::Mat_<float>(window_size, window_size);
            
            svr_templates[scale][view_id][ind].Response(area_of_interest, patch_expert_responses[ind]);
        }
    }
    
}

int PatchExperts::GetViewIdx(const cv::Vec3d& rvec, int scale) const
{
    cv::Mat R33;
    cv::Rodrigues(rvec, R33);
    R33.convertTo(R33, CV_32F);
    cv::Vec3f eulers = RotationMatrix2Euler(R33);
    
    int idx = 0;
    
    float dbest = 100;
    
    for(int i = 0; i < this->views[scale].size(); i++)
    {
        float v1 = eulers[0] - views[scale][i][0];
        float v2 = eulers[1] - views[scale][i][1];
        float v3 = eulers[2] - views[scale][i][2];
        
        float d = v1*v1 + v2*v2 + v3*v3;
        
        if(i == 0 || d < dbest)
        {
            dbest = d;
            idx = i;
        }
    }
    return idx;
}

int PatchExperts::GetScaleIdx(const cv::Size size, const cv::Mat lmks) const
{
    double min_x, min_y, max_x, max_y;
    cv::minMaxLoc(lmks.col(0), &min_x, &max_x);
    cv::minMaxLoc(lmks.col(1), &min_y, &max_y);
    const double estScale = ((max_x - min_x) / size.width + (max_y - min_y) / size.height) / 2;
    double minDist = double(INT_MAX);
    int bestScaleIdx = 0;
    for(int i = 1; i < scales.size(); i++)
    {
        if(abs(estScale - scales[i]) < minDist)
        {
            minDist = abs(estScale - scales[i]);
            bestScaleIdx = i;
        }
    }
    return bestScaleIdx;
}

bool PatchExperts::Read_SVR_patch_experts(string svr_expert_location, std::vector<cv::Vec3d>& views, std::vector<cv::Mat>& visibility, std::vector<std::vector<Multi_SVR_PatchExpert> >& patches, double& scale)
{
    ifstream patchesFile(svr_expert_location.c_str(), ios_base::in);
    
    if(patchesFile.is_open())
    {
        SkipComments(patchesFile);
        
        patchesFile >> scale;
        
        SkipComments(patchesFile);
        
        int numberViews;
        
        patchesFile >> numberViews;
        
        // read the visibility
        views.resize(numberViews);
        visibility.resize(numberViews);
        
        patches.resize(numberViews);
        
        SkipComments(patchesFile);
        
        // centers of each view (which view corresponds to which orientation)
        for(size_t i = 0; i < views.size(); i++)
        {
            cv::Mat center;
            ReadMat(patchesFile, center);
            center.copyTo(views[i]);
            views[i] = views[i] * M_PI / 180.0;
        }
        
        SkipComments(patchesFile);
        
        // the visibility of points for each of the views (which verts are visible at a specific view
        for(size_t i = 0; i < visibility.size(); i++)
        {
            ReadMat(patchesFile, visibility[i]);
        }
        
        int numberOfPoints = visibility[0].rows;
        
        SkipComments(patchesFile);
        
        // read the patches themselves
        for(size_t i = 0; i < patches.size(); i++)
        {
            // number of patches for each view
            patches[i].resize(numberOfPoints);
            // read in each patch
            for(int j = 0; j < numberOfPoints; j++)
            {
                patches[i][j].Read(patchesFile);
            }
        }
        
        cout << "Done" << endl;
        return true;
    }
    else
    {
        cout << "Can't find/open the patches file" << endl;
        return false;
    }
    return true;
}
bool PatchExperts::Read_CCNF_patch_experts(string patchesFileLocation, std::vector<cv::Vec3d>& centers, std::vector<cv::Mat>& visibility, std::vector<std::vector<CCNF_patch_expert> >& patches, double& patchScaling)
{
    
    ifstream patchesFile(patchesFileLocation.c_str(), ios::in | ios::binary);
    
    if(patchesFile.is_open())
    {
        patchesFile.read ((char*)&patchScaling, 8);
        
        int numberViews;
        patchesFile.read ((char*)&numberViews, 4);
        
        // read the visibility
        centers.resize(numberViews);
        visibility.resize(numberViews);
        
        patches.resize(numberViews);
        
        // centers of each view (which view corresponds to which orientation)
        for(size_t i = 0; i < centers.size(); i++)
        {
            cv::Mat center;
            ReadMatBin(patchesFile, center);
            center.copyTo(centers[i]);
            centers[i] = centers[i] * M_PI / 180.0;
        }
        
        // the visibility of points for each of the views (which verts are visible at a specific view
        for(size_t i = 0; i < visibility.size(); i++)
        {
            ReadMatBin(patchesFile, visibility[i]);
        }
        int numberOfPoints = visibility[0].rows;
        
        // Read the possible SigmaInvs (without beta), this will be followed by patch reading (this assumes all of them have the same type, and number of betas)
        int num_win_sizes;
        int num_sigma_comp;
        patchesFile.read ((char*)&num_win_sizes, 4);
        
        vector<int> windows;
        windows.resize(num_win_sizes);
        
        vector<vector<cv::Mat_<float> > > sigma_components;
        sigma_components.resize(num_win_sizes);
        
        for (int w=0; w < num_win_sizes; ++w)
        {
            patchesFile.read ((char*)&windows[w], 4);
            
            patchesFile.read ((char*)&num_sigma_comp, 4);
            
            sigma_components[w].resize(num_sigma_comp);
            
            for(int s=0; s < num_sigma_comp; ++s)
            {
                ReadMatBin(patchesFile, sigma_components[w][s]);
            }
        }
        
        this->sigma_components = sigma_components;
        
        // read the patches themselves
        for(size_t i = 0; i < patches.size(); i++)
        {
            // number of patches for each view
            patches[i].resize(numberOfPoints);
            // read in each patch
            for(int j = 0; j < numberOfPoints; j++)
            {
                patches[i][j].Read(patchesFile, windows, sigma_components);
            }
        }
        cout << "Done" << endl;
        return true;
    }
    else
    {
        cout << "Can't find/open the patches file" << endl;
        return false;
    }
}

std::vector<int> PatchExperts::Collect_visible_landmarks(vector<vector<cv::Mat> > visibilities, int scale, int view_id, int n)
{
    std::vector<int> vis_lmk;
    for (int i = 0; i < n; i++)
    {
        if (visibilities[scale][view_id].rows == n)
        {
            vis_lmk.push_back(i);
        }
    }
    return vis_lmk;
    
}

