//
//  LandmarkMapper.cpp
//  VSARDemo
//
//  Created by RongYuan on 9/3/18.
//  Copyright © 2018 VIPS. All rights reserved.
//

#include <stdio.h>
#include "LandmarkMapper.hpp"

const vector<vector<int>> LMK106_TO_LMK68 = {{1},{3},{5},{7},{9},{11},{13},{15},{17},{32},{30},{28},{26},{24},{22},{20},{18},{34},{35},{36},{37},{38},{43},{44},{45},{46},{47},{52},{53},{54},{55},{59},{60},{61},{66},{65},{67},{68,69},{69,70},{71},{72,73},{73,74},{77},{78,79},{79,80},{81},{82,83},{83,84},{87},{89},{88},{97},{93},{94},{92},{100},{101},{106},{104},{103},{90},{91},{98},{96},{95},{99},{105},{102}};

const vector<vector<int>> LMK106_TO_LMK5 = {{61}, {67}, {81}, {87}, {92}};

const vector<vector<int>> LMK106_TO_LMK76 = {{1},{3},{5},{7},{9},{11},{13},{15},{17},{32},{30},{28},{26},{24},{22},{20},{18},{34},{35},{36},{37},{38},{43},{44},{45},{46},{47},{52},{53},{54},{55},{59},{60},{61},{66},{65},{67},{68,69},{69,70},{71},{72,73},{73,74},{77},{78,79},{79,80},{81},{82,83},{83,84},{87},{89},{88},{97},{93},{94},{92},{100},{101},{106},{104},{103},{90},{91},{98},{96},{95},{99},{105},{102}, {39}, {40}, {41}, {42}, {48}, {49}, {50}, {51}};

cv::Mat mapFrom106Lmks(const cv::Mat &lmk106, const int n)
{
    int dim = lmk106.cols;
    const vector<vector<int>>* ptr = NULL;
    switch (n) {
        case 76:
            ptr = &LMK106_TO_LMK76;
            break;
        case 68:
            ptr = &LMK106_TO_LMK68;
            break;
        case 5:
            ptr = &LMK106_TO_LMK5;
            break;
        default:
            break;
    }
    cv::Mat output(n, dim, CV_32F);
    if(ptr)
    {
        for(int i = 0; i < n; i++)
        {
            if(ptr->at(i).size() == 1)
                lmk106.row(ptr->at(i)[0] - 1).copyTo(output.row(i));
            else if(ptr->at(i).size() == 2)
                output.row(i) = (lmk106.row(ptr->at(i)[0] - 1) + lmk106.row(ptr->at(i)[1] - 1)) / 2;
        }
    }
    return output.clone();
}


vector<cv::Point2f> mapFrom106Lmks(const vector<cv::Point2f> &lmk106, const int n)
{
    const vector<vector<int>>* ptr = NULL;
    switch (n) {
        case 76:
            ptr = &LMK106_TO_LMK76;
            break;
        case 68:
            ptr = &LMK106_TO_LMK68;
            break;
        case 5:
            ptr = &LMK106_TO_LMK5;
            break;
        default:
            break;
    }
    vector<cv::Point2f> output(n, cv::Point(0, 0));
    if(ptr)
    {
        for(int i = 0; i < n; i++)
        {
            if(ptr->at(i).size() == 1)
                output[i] = lmk106[ptr->at(i)[0] - 1];
            else if(ptr->at(i).size() == 2)
                output[i] = (lmk106[ptr->at(i)[0] - 1] + lmk106[ptr->at(i)[1] - 1]) / 2;
        }
    }
    return output;
}
