#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <opencv2/opencv.hpp>
#include "rapidjson/document.h"         //https://github.com/Tencent/rapidjson
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/filereadstream.h"
#include "opencv_utils.h"


using namespace std;

struct MetaData {
public:
    int infer_size[2];  // h w
    int image_size[2];  // h w
};


/**
 * 获取json配置文件
 * @param json_path 配置文件路径
 * @return
 */
MetaData getJson(const string& json_path);


/**
 * 获取文件夹下全部图片的绝对路径
 *
 * @param path          图片文件夹路径
 * @return result       全部图片绝对路径列表
 */
vector<cv::String> getImagePaths(string& path);



/**
 * 读取图像 BGR2RGB
 *
 * @param path  图片路径
 * @return      图片
 */
cv::Mat readImage(string& path);


/**
 * 保存图片和分数
 *
 * @param mixed_image_with_label 混合后的图片
 * @param image_path 输入图片的路径
 * @param save_dir   保存的路径
 */
void saveImages(vector<cv::Mat>& images, cv::String& image_path, string& save_dir);


/**
 * 图片预处理
 * @param image 预处理图片
 * @return      经过预处理的图片
 */
cv::Mat pre_process(cv::Mat& image, MetaData& meta);


/**
 * 叠加图片
 *
 * @param anomaly_map   混合后的图片
 * @param origin_image  原始图片
 * @return result       叠加后的图像
 */
cv::Mat superimposeAnomalyMap(cv::Mat& anomaly_map, cv::Mat& origin_image);


/**
 * 计算mask
 *
 * @param anomaly_map 热力图
 * @param threshold   二值化阈值
 * @param kernel_size 开操作kernel_size
 * @return mask
 */
cv::Mat compute_mask(cv::Mat& anomaly_map, float threshold=0.5, int kernel_size=1);


/**
 * 计算mask边界并混合到原图
 *
 * @param mask  mask
 * @param image 原图
 * @return      混合mask边界的原图
 */
cv::Mat gen_mask_border(cv::Mat& mask, cv::Mat& image);


/**
 * 生成mask,mask边缘,热力图和原图的叠加
 *
 * @param image        原图
 * @param anomaly_map  热力图
 * @param threshold    热力图二值化阈值
 * @return
 */
vector<cv::Mat> gen_images(cv::Mat& image, cv::Mat& anomaly_map, float threshold=0.5);