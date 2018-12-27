///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt
//
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////

#include "SVR_patch_expert.hpp"

#include "CLM_utils.hpp"

void Multi_SVR_PatchExpert::Read(ifstream &stream)
{
    // A sanity check when reading patch experts
    int type;
    stream >> type;
    assert(type == 3);
    
    // The number of patch experts for this view (with different modalities)
    int number_modalities;
    
    stream >> width >> height >> number_modalities;
    
    svr_patch_experts.resize(number_modalities);
    for(int i = 0; i < number_modalities; i++)
        svr_patch_experts[i].Read(stream);
    
}
void Multi_SVR_PatchExpert::Response(const cv::Mat &roi, cv::Mat &response)
{
    int response_height = roi.rows - height + 1;
    int response_width = roi.cols - width + 1;
    
    if(response.rows != response_height || response.cols != response_width)
        response = cv::Mat(response_height, response_width, CV_32F);
    
    // For the purposes of the experiment only use the response of normal intensity, for fair comparison
    
    if(svr_patch_experts.size() == 1)
    {
        svr_patch_experts[0].Response(roi, response);
    }
    else
    {
        // responses from multiple patch experts these can be gradients, LBPs etc.
        response.setTo(1.0);
        
        cv::Mat modality_resp(response_height, response_width, CV_32F);
        
        for(size_t i = 0; i < svr_patch_experts.size(); i++)
        {
            svr_patch_experts[i].Response(roi, modality_resp);
            response = response.mul(modality_resp);
        }
        
    }
}

void SVR_PatchExpert::Read(ifstream &stream)
{
    // A sanity check when reading patch experts
    int read_type;
    
    stream >> read_type;
    assert(read_type == 2);
    
    stream >> type >> confidence >> scaling >> bias;
    ReadMat(stream, weights);
    
    // OpenCV and Matlab matrix cardinality is different, hence the transpose
    weights = weights.t();
    
}

void SVR_PatchExpert::Response(const cv::Mat &roi, cv::Mat &response)
{
    int response_height = roi.rows - weights.rows + 1;
    int response_width = roi.cols - weights.cols + 1;
    
    // the patch area on which we will calculate reponses
    cv::Mat normalised_roi;
    
    if(response.rows != response_height || response.cols != response_width)
        response = cv::Mat(response_height, response_width, CV_32F);
    
    // If type is raw just normalise mean and standard deviation
    if(type == 0)
    {
        // Perform normalisation across whole patch
        cv::Scalar mean;
        cv::Scalar std;
        
        cv::meanStdDev(roi, mean, std);
        // Avoid division by zero
        if(std[0] == 0)
        {
            std[0] = 1;
        }
        normalised_roi = (roi - mean[0]) / std[0];
    }
    // If type is gradient, perform the image gradient computation
    else if(type == 1)
    {
        Grad(roi, normalised_roi);
    }
    else
    {
        printf("ERROR(%s,%d): Unsupported patch type %d!\n", __FILE__,__LINE__, type);
        abort();
    }
    
    cv::Mat svr_response;
    
    // The empty matrix as we don't pass precomputed dft's of image
    cv::Mat_<double> empty_matrix_0(0,0,0.0);
    cv::Mat_<float> empty_matrix_1(0,0,0.0);
    cv::Mat_<float> empty_matrix_2(0,0,0.0);
    
    // Efficient calc of patch expert SVR response across the area of interest
    matchTemplate_m(normalised_roi, empty_matrix_0, empty_matrix_1, empty_matrix_2, weights, weights_dfts, svr_response, CV_TM_CCOEFF_NORMED);
    response = cv::Mat(svr_response.rows, svr_response.cols, CV_32F);
    cv::MatIterator_<float> p = response.begin<float>();
    
    cv::MatIterator_<float> q1 = svr_response.begin<float>(); // respone for each pixel
    cv::MatIterator_<float> q2 = svr_response.end<float>();
    
    while(q1 != q2)
    {
        // the SVR response passed into logistic regressor
        *p++ = 1.0/(1.0 + exp( -(*q1++ * scaling + bias )));
    }
}
