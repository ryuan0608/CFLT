//
//  PDM.cpp
//
//  Created by RongYuan on 7/23/18.
//

#include "PDM.hpp"
#include "CLM_utils.hpp"
#include "LandmarkMapper.hpp"

int PDM::nDim() const
{
    return K;
}
int PDM::nPts() const
{
    return N;
}
cv::Mat PDM::mean() const
{
    return M.clone();
}
cv::Mat PDM::evs() const
{
    return E.clone();
}
cv::Mat PDM::bases() const
{
    return B.clone();
}
bool PDM::Read(string location)
{
    ifstream pdmLoc(location, ios_base::in);
    if (!pdmLoc.is_open())
    {
        return false;
    }
    
    SkipComments(pdmLoc);
    
    // Reading mean values
    cv::Mat_<double> mean_shape_d;
    ReadMat(pdmLoc, mean_shape_d);
    
    SkipComments(pdmLoc);
    
    // Reading principal components
    cv::Mat_<double> princ_comp_d;
    ReadMat(pdmLoc, princ_comp_d);
    
    
    SkipComments(pdmLoc);
    
    // Reading eigenvalues
    cv::Mat_<double> eigen_values_d;
    ReadMat(pdmLoc, eigen_values_d);
    
    N = mean_shape_d.rows / 3;
    K = min(eigen_values_d.cols, PCA_DIM);
    
    B = cv::Mat(3*K, N, CV_32F);
    M = cv::Mat(3, N, CV_32F);
    E = cv::Mat(K, 1, CV_32F);
    
    // build bases in matrix format
    for(int k = 0; k < K; k++)
    {
//        float ev = float(eigen_values_d.at<double>(k));
        for(int i = 0; i < N; i++)
        {
            B.at<float>(3*k+0, i) = float(princ_comp_d.at<double>(i, k));
            B.at<float>(3*k+1, i) = float(princ_comp_d.at<double>(N+i, k));
            B.at<float>(3*k+2, i) = float(princ_comp_d.at<double>(2*N+i, k));
        }
    }
    // build mean in matrix format (3xN)
    for(int i = 0; i < N; i++)
    {
        M.at<float>(0, i) = float(mean_shape_d.at<double>(i));
        M.at<float>(1, i) = float(mean_shape_d.at<double>(N+i));
        M.at<float>(2, i) = float(mean_shape_d.at<double>(2*N+i));
    }
    
    for(int i = 0; i < K; i++)
    {
        E.at<float>(i) = float(eigen_values_d.at<double>(i));
    }
    
    params = cv::Mat::zeros(K, 1, CV_32F);
    
    // initialize some auxiliray variables
    shape = cv::Mat::zeros(3, N, CV_32F);
    updateShape();
    
    return true;
}
cv::Mat PDM::calc2DShape() const
{
    vector<cv::Point2f> shape2d;
    vector<cv::Point3f> shape3d;
    for(int i = 0; i < N; i++)
        shape3d.push_back(cv::Point3f(shape.col(i)));
    cv::projectPoints(shape3d, rvec, tvec, intrinsic(), cv::Mat(), shape2d);
    cv::Mat res(shape2d.size(), 2, CV_32F);
    for(int i = 0; i < shape2d.size(); i++)
    {
        res.at<float>(i, 0) = shape2d[i].x;
        res.at<float>(i, 1) = shape2d[i].y;
    }
    return res.clone();
}
cv::Vec3d PDM::getRvec() const
{
    return rvec;
}
cv::Vec3d PDM::getTvec() const
{
    return tvec;
}
void PDM::setRvec(const cv::Vec3d &_rvec)
{
    rvec = _rvec;
    cv::Rodrigues(rvec, R33);
    R33.convertTo(R33, CV_32F);
}
void PDM::setTvec(const cv::Vec3d &_tvec)
{
    tvec = _tvec;
    for(int i = 0; i < 3; i++)
        T3.at<float>(i) = tvec[i];
}
void PDM::getParams(cv::Mat &output) const
{
    params.copyTo(output);
}
void PDM::setParams(const cv::Mat &_params)
{
    params = _params.clone();
    updateShape();
}
void PDM::getShape(cv::Mat &output) const
{
    shape.copyTo(output);
}
void PDM::residual(cv::Mat &output) const
{
    cv::Mat shape2D = calc2DShape();
    cv::Mat r = shape2D - W;
    output = r.reshape(1, 2*N);
}
float PDM::err() const
{
    cv::Mat res;
    residual(res);
    return cv::norm(res);
}
// Face PCA setters and getters
void PDM::updateShape()
{
    shape = M.clone();
    for(int k = 0; k < K; k++)
    {
        shape += B.rowRange(k*3, k*3+3) * params.at<float>(k);
    }
}

PDM::PDM()
{
    // read in raw model info
    R33 = cv::Mat::eye(3, 3, CV_32F);
    T3 = cv::Mat::zeros(3, 1, CV_32F);
    setRvec(cv::Vec3d(0, 0, 0));
    setTvec(cv::Vec3d(0, 0, 0));
}

void PDM::jacobianWeights(cv::Mat &output) const
{
    cv::Mat intr = intrinsic();
    float focal = intr.at<float>(0, 0);
    cv::Mat jacobian(2*N, K, CV_32F);
    float r11 = R33.at<float>(0, 0);
    float r12 = R33.at<float>(0, 1);
    float r13 = R33.at<float>(0, 2);
    float r21 = R33.at<float>(1, 0);
    float r22 = R33.at<float>(1, 1);
    float r23 = R33.at<float>(1, 2);
    float r31 = R33.at<float>(2, 0);
    float r32 = R33.at<float>(2, 1);
    float r33 = R33.at<float>(2, 2);
    float tx = T3.at<float>(0);
    float ty = T3.at<float>(1);
    float tz = T3.at<float>(2);
    
    for(int j = 0; j < K; j++)
    {
        for(int i = 0; i < N; i++)
        {
            float xi = shape.at<float>(0, i);
            float yi = shape.at<float>(1, i);
            float zi = shape.at<float>(2, i);
            
            cv::Mat Bij;
            B.rowRange(j*3, j*3+3).col(i).copyTo(Bij);
            float Bijx = Bij.at<float>(0);
            float Bijy = Bij.at<float>(1);
            float Bijz = Bij.at<float>(2);
            float term7 = (r11*Bijx+r12*Bijy+r13*Bijz)*(r31*xi+r32*yi+r33*zi+tz)
            -(r11*xi+r12*yi+r13*zi+tx)*(r31*Bijx+r32*Bijy+r33*Bijz);
            float term9 = (r21*Bijx+r22*Bijy+r23*Bijz)*(r31*xi+r32*yi+r33*zi+tz)
            -(r21*xi+r22*yi+r23*zi+ty)*(r31*Bijx+r32*Bijy+r33*Bijz);
            float term8 = (r31*xi+r32*yi+r33*zi+tz)*(r31*xi+r32*yi+r33*zi+tz);
            
            
            float d1 = focal * term7 / term8;
            float d2 = focal * term9 / term8;
            
            jacobian.at<float>(2*i, j) = d1;
            jacobian.at<float>(2*i+1, j) = d2;
        }
    }
    jacobian.copyTo(output);
}

void PDM::jacobianRT(cv::Mat &output) const
{
    cv::Mat intr = intrinsic();
    float focal = intr.at<float>(0, 0);
    cv::Mat jacobian(2*N, 6, CV_32F); // 3 for euler angles, 3 for translation vector
    cv::Vec3d rvec = getRvec();
    cv::Mat JR;
    cv::Mat _R33;
    cv::Rodrigues(rvec, _R33, JR);
    JR.convertTo(JR, CV_32F);
    float JR11x = JR.at<float>(0, 0);
    float JR12x = JR.at<float>(0, 1);
    float JR13x = JR.at<float>(0, 2);
    float JR21x = JR.at<float>(0, 3);
    float JR22x = JR.at<float>(0, 4);
    float JR23x = JR.at<float>(0, 5);
    float JR31x = JR.at<float>(0, 6);
    float JR32x = JR.at<float>(0, 7);
    float JR33x = JR.at<float>(0, 8);
    float JR11y = JR.at<float>(1, 0);
    float JR12y = JR.at<float>(1, 1);
    float JR13y = JR.at<float>(1, 2);
    float JR21y = JR.at<float>(1, 3);
    float JR22y = JR.at<float>(1, 4);
    float JR23y = JR.at<float>(1, 5);
    float JR31y = JR.at<float>(1, 6);
    float JR32y = JR.at<float>(1, 7);
    float JR33y = JR.at<float>(1, 8);
    float JR11z = JR.at<float>(2, 0);
    float JR12z = JR.at<float>(2, 1);
    float JR13z = JR.at<float>(2, 2);
    float JR21z = JR.at<float>(2, 3);
    float JR22z = JR.at<float>(2, 4);
    float JR23z = JR.at<float>(2, 5);
    float JR31z = JR.at<float>(2, 6);
    float JR32z = JR.at<float>(2, 7);
    float JR33z = JR.at<float>(2, 8);
    float r11 = R33.at<float>(0, 0);
    float r12 = R33.at<float>(0, 1);
    float r13 = R33.at<float>(0, 2);
    float r21 = R33.at<float>(1, 0);
    float r22 = R33.at<float>(1, 1);
    float r23 = R33.at<float>(1, 2);
    float r31 = R33.at<float>(2, 0);
    float r32 = R33.at<float>(2, 1);
    float r33 = R33.at<float>(2, 2);
    float tx = T3.at<float>(0);
    float ty = T3.at<float>(1);
    float tz = T3.at<float>(2);
    for(int i = 0; i < N; i++)
    {
        float xi = shape.at<float>(0, i);
        float yi = shape.at<float>(1, i);
        float zi = shape.at<float>(2, i);
        // alpha, u
        float term2 = (xi*JR11x+yi*JR12x+zi*JR13x)*(r31*xi+r32*yi+r33*zi+tz)
        - (r11*xi+r12*yi+r13*zi+tx)*(xi*JR31x+yi*JR32x+zi*JR33x);
        float term3 = (r31*xi+r32*yi+r33*zi+tz)*(r31*xi+r32*yi+r33*zi+tz);
        jacobian.at<float>(i*2, 0) = focal * term2 / term3;
        // beta, u
        float term5 = (xi*JR11y+yi*JR12y+zi*JR13y)*(r31*xi+r32*yi+r33*zi+tz)
        - (r11*xi+r12*yi+r13*zi+tx)*(xi*JR31y+yi*JR32y+zi*JR33y);
        float term6 = term3;
        jacobian.at<float>(i*2, 1) = focal * term5 / term6;
        // theta, u
        float term8 = (xi*JR11z+yi*JR12z+zi*JR13z)*(r31*xi+r32*yi+r33*zi+tz)
        - (r11*xi+r12*yi+r13*zi+tx)*(xi*JR31z+yi*JR32z+zi*JR33z);
        float term9 = term3;
        jacobian.at<float>(i*2, 2) = focal * term8 / term9;
        // alpha, v
        float term11 = (xi*JR21x+yi*JR22x+zi*JR23x)*(r31*xi+r32*yi+r33*zi+tz)
        - (r21*xi+r22*yi+r23*zi+ty)*(xi*JR31x+yi*JR32x+zi*JR33x);
        float term12 = term3;
        jacobian.at<float>(i*2+1, 0) = focal * term11 / term12;
        // beta, v
        float term14 = (xi*JR21y+yi*JR22y+zi*JR23y)*(r31*xi+r32*yi+r33*zi+tz)
        - (r21*xi+r22*yi+r23*zi+ty)*(xi*JR31y+yi*JR32y+zi*JR33y);
        float term15 = term3;
        jacobian.at<float>(i*2+1, 1) = focal * term14 / term15;
        // theta, v
        float term17 = (xi*JR21z+yi*JR22z+zi*JR23z)*(r31*xi+r32*yi+r33*zi+tz)
        - (r21*xi+r22*yi+r23*zi+ty)*(xi*JR31z+yi*JR32z+zi*JR33z);
        float term18 = term3;
        jacobian.at<float>(i*2+1, 2) = focal * term17 / term18;
        // tx, ty, tz, u
        float ri_u_tx = focal / (r31*xi+r32*yi+r33*zi+tz);
        float ri_u_ty = 0;
        float ri_u_tz = -focal * (r11*xi+r12*yi+r13*zi+tx) / ((r31*xi+r32*yi+r33*zi+tz)*(r31*xi+r32*yi+r33*zi+tz));
        jacobian.at<float>(i*2, 3) = ri_u_tx;
        jacobian.at<float>(i*2, 4) = ri_u_ty;
        jacobian.at<float>(i*2, 5) = ri_u_tz;
        // tx, ty, tz, v
        float ri_v_tx = 0;
        float ri_v_ty = focal / (r31*xi+r32*yi+r33*zi+tz);
        float ri_v_tz = -focal * (r21*xi+r22*yi+r23*zi+ty) / ((r31*xi+r32*yi+r33*zi+tz)*(r31*xi+r32*yi+r33*zi+tz));
        jacobian.at<float>(i*2+1, 3) = ri_v_tx;
        jacobian.at<float>(i*2+1, 4) = ri_v_ty;
        jacobian.at<float>(i*2+1, 5) = ri_v_tz;
    }
    jacobian.copyTo(output);
}
void PDM::jacobian(cv::Mat &output) const
{
    cv::Mat jacWeights, jacRT;
    jacobianWeights(jacWeights);
    jacobianRT(jacRT);
    cv::Mat jac;
    cv::hconcat(jacRT, jacWeights, output);
}

void PDM::calcParams(vector<cv::Point2f> lmks2d)
{
    // initial local parameter guess
    cv::Mat _params = cv::Mat::zeros(K, 1, CV_32F);
    setParams(_params);
    W = cv::Mat::zeros(N, 2, CV_32F);
    vector<int> visibility(N, 0);
    vector<cv::Point2f> visibleLmks2d;
    vector<cv::Point3f> visibleLmks3d;
    int visiCount = 0;
    for(int i = 0; i < N; i++)
    {
        // invisible
        if(lmks2d[i].x == 0 && lmks2d[i].y == 0)
            continue;
        W.at<float>(i, 0) = lmks2d[i].x;
        W.at<float>(i, 1) = lmks2d[i].y;
        visibleLmks2d.push_back(lmks2d[i]);
        visibleLmks3d.push_back(cv::Point3f(shape.col(i)));
        visibility[i] = 1;
        ++visiCount;
    }
    
    cv::Mat regularisations = cv::Mat::zeros(1, 6+K, CV_32F);
    const float regFactor = 1;
    
    cv::Mat(regFactor / E.t()).copyTo(regularisations(cv::Rect(6, 0, K, 1)));
    regularisations = cv::Mat::diag(regularisations.t());
    
    cv::Mat weightMatrix = cv::Mat::eye(2*N, 2*N, CV_32F);
    
    // initial global parameter guess
    cv::Vec3d _rvec, _tvec;
    cv::Mat _shape;
    getShape(_shape);
    _shape = _shape.t();
    cv::Mat _lmks;
    W.copyTo(_lmks);
    cv::solvePnP(visibleLmks3d, visibleLmks2d, intrinsic(), cv::Mat(), _rvec, _tvec);
    setRvec(_rvec);
    setTvec(_tvec);
    float currErr = -1;
    
    cv::Mat J, r, J_visi(2*visiCount, 6+K, CV_32F), r_visi(2*visiCount, 1, CV_32F);
    cv::Mat Hessian;
    cv::Mat delta;
    cv::Mat fullParamsList(6+K, 1, CV_32F);
    for(size_t i = 0; i < 1000; i++)
    {
        fullParamsList.at<float>(0) = _rvec[0];
        fullParamsList.at<float>(1) = _rvec[1];
        fullParamsList.at<float>(2) = _rvec[2];
        fullParamsList.at<float>(3) = _tvec[0];
        fullParamsList.at<float>(4) = _tvec[1];
        fullParamsList.at<float>(5) = _tvec[2];
        for(int k = 0; k < K; k++)
            fullParamsList.at<float>(6+k) = _params.at<float>(k);
        
        jacobian(J);
        residual(r);
        // set invisible units to zero
        for(int i = 0, k = 0; i < visibility.size(); i++)
        {
            if(visibility[i])
            {
                J.rowRange(2*i, 2*i+2).copyTo(J_visi.rowRange(2*k, 2*k+2));
                r.rowRange(2*i, 2*i+2).copyTo(r_visi.rowRange(2*k, 2*k+2));
                ++k;
            }
        }
        
        Hessian = J_visi.t() * J_visi + regularisations;
        cv::solve(Hessian, J_visi.t() * r_visi - regularisations * fullParamsList, delta);
        delta = 0.75 * delta;
        
        // update
        _rvec[0] -= delta.at<float>(0);
        _rvec[1] -= delta.at<float>(1);
        _rvec[2] -= delta.at<float>(2);
        _tvec[0] -= delta.at<float>(3);
        _tvec[1] -= delta.at<float>(4);
        _tvec[2] -= delta.at<float>(5);
        for(int k = 0; k < K; k++)
            _params.at<float>(k) -= delta.at<float>(6+k);
        
        // might not be a good idea to clamp to 3 sigma here
        // try it out in the future work
        
        setRvec(_rvec);
        setTvec(_tvec);
        setParams(_params);
        
        residual(r);
        
        // set invisible units to zero
        for(int i = 0, k = 0; i < visibility.size(); i++)
        {
            if(visibility[i])
            {
                r.rowRange(2*i, 2*i+2).copyTo(r_visi.rowRange(2*k, 2*k+2));
                ++k;
            }
        }
        
        float error = cv::norm(r_visi);
        if(currErr > 0 && 0.999 * currErr < error)
            break;
        
        currErr = error;
    }
    
    
}
