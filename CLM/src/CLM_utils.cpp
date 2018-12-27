//
//  CLM_utils.hpp
//  Pods
//
//  Created by RongYuan on 7/20/18.
//

#include "CLM_utils.hpp"

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;

string OPEN_FACE_DIR = "./";


vector<string> getDirectoryFiles(const string& dir)
{
    vector<string> files;
    shared_ptr<DIR> directory_ptr(opendir(dir.c_str()), [](DIR* dir){ dir && closedir(dir); });
    struct dirent *dirent_ptr;
    if (!directory_ptr) {
        std::cout << "Error opening : " << std::strerror(errno) << dir << std::endl;
        return files;
    }
    
    while ((dirent_ptr = readdir(directory_ptr.get())) != nullptr) {
        files.push_back(std::string(dirent_ptr->d_name));
    }
    return files;
}

// Skipping lines that start with # (together with empty lines)
void SkipComments(std::ifstream& stream)
{
    while (stream.peek() == '#' || stream.peek() == '\n' || stream.peek() == ' ' || stream.peek() == '\r')
    {
        std::string skipped;
        std::getline(stream, skipped);
    }
}

// Reading in a matrix from a stream
void ReadMat(std::ifstream& stream, cv::Mat &output_mat)
{
    // Read in the number of rows, columns and the data type
    int row, col, type;
    
    stream >> row >> col >> type;
    
    output_mat = cv::Mat(row, col, type);
    
    switch (output_mat.type())
    {
        case CV_64FC1:
        {
            cv::MatIterator_<double> begin_it = output_mat.begin<double>();
            cv::MatIterator_<double> end_it = output_mat.end<double>();
            
            while (begin_it != end_it)
            {
                stream >> *begin_it++;
            }
        }
            break;
        case CV_32FC1:
        {
            cv::MatIterator_<float> begin_it = output_mat.begin<float>();
            cv::MatIterator_<float> end_it = output_mat.end<float>();
            
            while (begin_it != end_it)
            {
                stream >> *begin_it++;
            }
        }
            break;
        case CV_32SC1:
        {
            cv::MatIterator_<int> begin_it = output_mat.begin<int>();
            cv::MatIterator_<int> end_it = output_mat.end<int>();
            while (begin_it != end_it)
            {
                stream >> *begin_it++;
            }
        }
            break;
        case CV_8UC1:
        {
            cv::MatIterator_<uchar> begin_it = output_mat.begin<uchar>();
            cv::MatIterator_<uchar> end_it = output_mat.end<uchar>();
            while (begin_it != end_it)
            {
                stream >> *begin_it++;
            }
        }
            break;
        default:
            printf("ERROR(%s,%d) : Unsupported Matrix type %d!\n", __FILE__, __LINE__, output_mat.type()); abort();
    }
}

void ReadMatBin(std::ifstream& stream, cv::Mat &output_mat)
{
    // Read in the number of rows, columns and the data type
    int row, col, type;
    
    stream.read((char*)&row, 4);
    stream.read((char*)&col, 4);
    stream.read((char*)&type, 4);
    
    output_mat = cv::Mat(row, col, type);
    int size = output_mat.rows * output_mat.cols * output_mat.elemSize();
    stream.read((char *)output_mat.data, size);
    
}

cv::Mat AlignShapesKabsch2D_f(const cv::Mat_<float>& align_from, const cv::Mat_<float>& align_to)
{
    
    cv::SVD svd(align_from.t() * align_to);
    
    // make sure no reflection is there
    // corr ensures that we do only rotaitons and not reflections
    float d = cv::determinant(svd.vt.t() * svd.u.t());
    
    cv::Matx22f corr = cv::Matx22f::eye();
    if (d > 0)
    {
        corr(1, 1) = 1;
    }
    else
    {
        corr(1, 1) = -1;
    }
    
    cv::Mat R;
    cv::Mat(svd.vt.t()*cv::Mat(corr)*svd.u.t()).copyTo(R);
    
    return R;
}

void Grad(const cv::Mat& im, cv::Mat& grad)
{
    int x,y,h = im.rows,w = im.cols;
    float vx,vy;
    
    // Initialise the gradient
    grad.create(im.size(), CV_32F);
    grad.setTo(0.0f);
    
    cv::MatIterator_<float> gp  = grad.begin<float>() + w+1;
    cv::MatConstIterator_<float> px1 = im.begin<float>()   + w+2;
    cv::MatConstIterator_<float> px2 = im.begin<float>()   + w;
    cv::MatConstIterator_<float> py1 = im.begin<float>()   + 2*w+1;
    cv::MatConstIterator_<float> py2 = im.begin<float>()   + 1;
    
    for(y = 1; y < h-1; y++)
    {
        for(x = 1; x < w-1; x++)
        {
            vx = *px1++ - *px2++;
            vy = *py1++ - *py2++;
            *gp++ = vx*vx + vy*vy;
        }
        px1 += 2;
        px2 += 2;
        py1 += 2;
        py2 += 2;
        gp += 2;
    }
    
}

void crossCorr_m(const cv::Mat& img, cv::Mat_<double>& img_dft, const cv::Mat_<float>& _templ, map<int, cv::Mat_<double> >& _templ_dfts, cv::Mat& corr)
{
    // Our model will always be under min block size so can ignore this
    //const double blockScale = 4.5;
    //const int minBlockSize = 256;
    
    int maxDepth = CV_64F;
    
    cv::Size dftsize;
    
    dftsize.width = cv::getOptimalDFTSize(corr.cols + _templ.cols - 1);
    dftsize.height = cv::getOptimalDFTSize(corr.rows + _templ.rows - 1);
    
    // Compute block size
    cv::Size blocksize;
    blocksize.width = dftsize.width - _templ.cols + 1;
    blocksize.width = MIN(blocksize.width, corr.cols);
    blocksize.height = dftsize.height - _templ.rows + 1;
    blocksize.height = MIN(blocksize.height, corr.rows);
    
    cv::Mat_<double> dftTempl;
    
    // if this has not been precomputed, precompute it, otherwise use it
    if (_templ_dfts.find(dftsize.width) == _templ_dfts.end())
    {
        dftTempl.create(dftsize.height, dftsize.width);
        
        cv::Mat src = _templ;
        
        cv::Mat_<double> dst(dftTempl, cv::Rect(0, 0, dftsize.width, dftsize.height));
        
        cv::Mat_<double> dst1(dftTempl, cv::Rect(0, 0, _templ.cols, _templ.rows));
        
        if (dst1.data != src.data)
            src.convertTo(dst1, dst1.depth());
        
        if (dst.cols > _templ.cols)
        {
            cv::Mat_<double> part(dst, cv::Range(0, _templ.rows), cv::Range(_templ.cols, dst.cols));
            part.setTo(0);
        }
        
        // Perform DFT of the template
        dft(dst, dst, 0, _templ.rows);
        
        _templ_dfts[dftsize.width] = dftTempl;
        
    }
    else
    {
        // use the precomputed version
        dftTempl = _templ_dfts.find(dftsize.width)->second;
    }
    
    cv::Size bsz(std::min(blocksize.width, corr.cols), std::min(blocksize.height, corr.rows));
    cv::Mat src;
    
    cv::Mat cdst(corr, cv::Rect(0, 0, bsz.width, bsz.height));
    
    cv::Mat_<double> dftImg;
    
    if (img_dft.empty())
    {
        dftImg.create(dftsize);
        dftImg.setTo(0.0);
        
        cv::Size dsz(bsz.width + _templ.cols - 1, bsz.height + _templ.rows - 1);
        
        int x2 = std::min(img.cols, dsz.width);
        int y2 = std::min(img.rows, dsz.height);
        
        cv::Mat src0(img, cv::Range(0, y2), cv::Range(0, x2));
        cv::Mat dst(dftImg, cv::Rect(0, 0, dsz.width, dsz.height));
        cv::Mat dst1(dftImg, cv::Rect(0, 0, x2, y2));
        
        src = src0;
        
        if (dst1.data != src.data)
            src.convertTo(dst1, dst1.depth());
        
        dft(dftImg, dftImg, 0, dsz.height);
        img_dft = dftImg.clone();
    }
    
    cv::Mat dftTempl1(dftTempl, cv::Rect(0, 0, dftsize.width, dftsize.height));
    cv::mulSpectrums(img_dft, dftTempl1, dftImg, 0, true);
    cv::dft(dftImg, dftImg, cv::DFT_INVERSE + cv::DFT_SCALE, bsz.height);
    
    src = dftImg(cv::Rect(0, 0, bsz.width, bsz.height));
    
    src.convertTo(cdst, CV_32F);
    
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

void matchTemplate_m(const cv::Mat& input_img, cv::Mat_<double>& img_dft, cv::Mat& _integral_img, cv::Mat& _integral_img_sq, const cv::Mat_<float>&  templ, map<int, cv::Mat_<double> >& templ_dfts, cv::Mat& result, int method)
{
    
    int numType = method == CV_TM_CCORR || method == CV_TM_CCORR_NORMED ? 0 :
    method == CV_TM_CCOEFF || method == CV_TM_CCOEFF_NORMED ? 1 : 2;
    bool isNormed = method == CV_TM_CCORR_NORMED ||
    method == CV_TM_SQDIFF_NORMED ||
    method == CV_TM_CCOEFF_NORMED;
    
    // Assume result is defined properly
    if (result.empty())
    {
        result = cv::Mat(input_img.rows - templ.rows + 1, input_img.cols - templ.cols + 1, CV_32F);
    }
    crossCorr_m(input_img, img_dft, templ, templ_dfts, result);
    
    if (method == CV_TM_CCORR)
        return;
    
    double invArea = 1. / ((double)templ.rows * templ.cols);
    
    cv::Mat sum, sqsum;
    cv::Scalar templMean, templSdv;
    double *q0 = 0, *q1 = 0, *q2 = 0, *q3 = 0;
    double templNorm = 0, templSum2 = 0;
    
    if (method == CV_TM_CCOEFF)
    {
        // If it has not been precomputed compute it now
        if (_integral_img.empty())
        {
            integral(input_img, _integral_img, CV_64F);
        }
        sum = _integral_img;
        
        templMean = cv::mean(templ);
    }
    else
    {
        // If it has not been precomputed compute it now
        if (_integral_img.empty())
        {
            integral(input_img, _integral_img, _integral_img_sq, CV_64F);
        }
        
        sum = _integral_img;
        sqsum = _integral_img_sq;
        
        meanStdDev(templ, templMean, templSdv);
        
        templNorm = templSdv[0] * templSdv[0] + templSdv[1] * templSdv[1] + templSdv[2] * templSdv[2] + templSdv[3] * templSdv[3];
        
        if (templNorm < DBL_EPSILON && method == CV_TM_CCOEFF_NORMED)
        {
            result.setTo(1.0);
            return;
        }
        
        templSum2 = templNorm + templMean[0] * templMean[0] + templMean[1] * templMean[1] + templMean[2] * templMean[2] + templMean[3] * templMean[3];
        
        if (numType != 1)
        {
            templMean = cv::Scalar::all(0);
            templNorm = templSum2;
        }
        
        templSum2 /= invArea;
        templNorm = std::sqrt(templNorm);
        templNorm /= std::sqrt(invArea); // care of accuracy here
        
        q0 = (double*)sqsum.data;
        q1 = q0 + templ.cols;
        q2 = (double*)(sqsum.data + templ.rows*sqsum.step);
        q3 = q2 + templ.cols;
    }
    
    double* p0 = (double*)sum.data;
    double* p1 = p0 + templ.cols;
    double* p2 = (double*)(sum.data + templ.rows*sum.step);
    double* p3 = p2 + templ.cols;
    
    int sumstep = sum.data ? (int)(sum.step / sizeof(double)) : 0;
    int sqstep = sqsum.data ? (int)(sqsum.step / sizeof(double)) : 0;
    
    int i, j;
    
    for (i = 0; i < result.rows; i++)
    {
        float* rrow = result.ptr<float>(i);
        int idx = i * sumstep;
        int idx2 = i * sqstep;
        
        for (j = 0; j < result.cols; j++, idx += 1, idx2 += 1)
        {
            //                cout << "normalised roi: " << endl;
            //                cout << input_img << endl;
            //                cout << "weights: " << endl;
            //                cout << templ << endl;
            double num = rrow[j], t;
            double wndMean2 = 0, wndSum2 = 0;
            //                cout << "num:" << endl;
            //                cout << num << endl;
            if (numType == 1)
            {
                t = p0[idx] - p1[idx] - p2[idx] + p3[idx];
                wndMean2 += t*t;
                num -= t*templMean[0];
                
                wndMean2 *= invArea;
            }
            //                cout << "num:" << endl;
            //                cout << num << endl;
            //                cout << "wndMean2:" << endl;
            //                cout << wndMean2 << endl;
            if (isNormed || numType == 2)
            {
                
                t = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2];
                wndSum2 += t;
                
                if (numType == 2)
                {
                    num = wndSum2 - 2 * num + templSum2;
                    num = MAX(num, 0.);
                }
            }
            //                cout << "wndSum2:" << endl;
            //                cout << wndSum2 << endl;
            //                cout << "templNorm:" << endl;
            //                cout << templNorm << endl;
            if (isNormed)
            {
                t = std::sqrt(MAX(wndSum2 - wndMean2, 0))*templNorm;
                if (fabs(num) < t)
                    num /= t;
                else if (fabs(num) < t*1.125)
                    num = num > 0 ? 1 : -1;
                else
                    num = method != CV_TM_SQDIFF_NORMED ? 0 : 1;
            }
            //                cout << "num:" << endl;
            //                cout << num << endl;
            
            rrow[j] = (float)num;
        }
    }
}

cv::Mat AlignShapesWithScale_f(cv::Mat& src, cv::Mat dst)
{
    int n = src.rows;
    
    // First we mean normalise both src and dst
    float mean_src_x = cv::mean(src.col(0))[0];
    float mean_src_y = cv::mean(src.col(1))[0];
    
    float mean_dst_x = cv::mean(dst.col(0))[0];
    float mean_dst_y = cv::mean(dst.col(1))[0];
    
    cv::Mat_<float> src_mean_normed = src.clone();
    src_mean_normed.col(0) = src_mean_normed.col(0) - mean_src_x;
    src_mean_normed.col(1) = src_mean_normed.col(1) - mean_src_y;
    
    cv::Mat_<float> dst_mean_normed = dst.clone();
    dst_mean_normed.col(0) = dst_mean_normed.col(0) - mean_dst_x;
    dst_mean_normed.col(1) = dst_mean_normed.col(1) - mean_dst_y;
    
    // Find the scaling factor of each
    cv::Mat src_sq;
    cv::pow(src_mean_normed, 2, src_sq);
    
    cv::Mat dst_sq;
    cv::pow(dst_mean_normed, 2, dst_sq);
    
    float s_src = sqrt(cv::sum(src_sq)[0] / n);
    float s_dst = sqrt(cv::sum(dst_sq)[0] / n);
    
    src_mean_normed = src_mean_normed / s_src;
    dst_mean_normed = dst_mean_normed / s_dst;
    
    float s = s_dst / s_src;
    
    // Get the rotation
    cv::Mat R = AlignShapesKabsch2D_f(src_mean_normed, dst_mean_normed);
    
    cv::Mat A = s * R;
    
    cv::Mat_<float> aligned = (cv::Mat(cv::Mat(A) * src.t())).t();
    cv::Mat_<float> offset = dst - aligned;
    
    float t_x = cv::mean(offset.col(0))[0];
    float t_y = cv::mean(offset.col(1))[0];
    
    return A;
    
}

// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
cv::Vec3f RotationMatrix2Euler(const cv::Mat& rotation_matrix)
{
    float q0 = sqrt(1 + rotation_matrix.at<float>(0, 0) + rotation_matrix.at<float>(1, 1) + rotation_matrix.at<float>(2, 2)) / 2.0f;
    float q1 = (rotation_matrix.at<float>(2, 1) - rotation_matrix.at<float>(1, 2)) / (4.0f*q0);
    float q2 = (rotation_matrix.at<float>(0, 2) - rotation_matrix.at<float>(2, 0)) / (4.0f*q0);
    float q3 = (rotation_matrix.at<float>(1, 0) - rotation_matrix.at<float>(0, 1)) / (4.0f*q0);
    
    // Slower, but dealing with degenerate cases due to precision
    float t1 = 2.0f * (q0*q2 + q1*q3);
    if (t1 > 1) t1 = 1.0f;
    if (t1 < -1) t1 = -1.0f;
    
    float yaw = asin(t1);
    float pitch = atan2(2.0f * (q0*q1 - q2*q3), q0*q0 - q1*q1 - q2*q2 + q3*q3);
    float roll = atan2(2.0f * (q0*q3 - q1*q2), q0*q0 + q1*q1 - q2*q2 - q3*q3);
    
    return cv::Vec3f(pitch, yaw, roll);
}

cv::Mat intrinsic()
{
    cv::Mat intr = cv::Mat::eye(3, 3, CV_32F);
    // 初始化的固定值适合iOS但是不适用于android,因此需要根据图片的宽和高进行
    double fx = 520;
    double fy = 520;
    double cx = 200;
    double cy = 320;
    
    cx = 1.0 * 480.0 / 2.0;
    cy = 1.0 * (float)640.0 / 2.0;
    
    fx = 500 * ((float)480.0  / 480.0);
    
    fy = 500 * ((float)640.0 / 480.0);
    
    fx = (fx + fy) / 2.0;
    fy = fx;
    
    intr.at<float>(0, 0) = fx;
    intr.at<float>(1, 1) = fy;
    intr.at<float>(0, 2) = cx;
    intr.at<float>(1, 2) = cy;
    
    return intr.clone();
}
