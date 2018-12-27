#ifndef __SVR_PATCH_EXPERT_hpp_
#define __SVR_PATCH_EXPERT_hpp_

#include <fstream>
#include <map>
#include <opencv2/core/core.hpp>

using namespace std;

class SVR_PatchExpert
{
public:
    SVR_PatchExpert(){;}
    void Read(ifstream &stream);
    void Response(const cv::Mat &roi, cv::Mat &response);
    
    int type;
    double confidence;
    double  scaling;
    double  bias;
    cv::Mat weights;
    map<int, cv::Mat_<double> > weights_dfts;
};

class Multi_SVR_PatchExpert
{
public:
    Multi_SVR_PatchExpert(){;}
    
    void Read(ifstream &stream);
    void Response(const cv::Mat &roi, cv::Mat &response);
    
    int width, height;
    
    vector<SVR_PatchExpert> svr_patch_experts;
};

#endif
