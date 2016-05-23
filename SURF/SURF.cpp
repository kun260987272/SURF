#include <cstdio>
#include <iostream>
#include <opencv2/nonfree/nonfree.hpp>  
#include<opencv2/opencv.hpp>

using namespace cv;

int main()
{
	//读取图像
	Mat imgSrc1 = imread("asong1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgSrc2 = imread("asong2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	//使用SURF检测特征点
	int minHessian = 4000;

	SurfFeatureDetector detector(minHessian);

	std::vector<KeyPoint> keypoints1, keypoints2;

	detector.detect(imgSrc1, keypoints1);
	detector.detect(imgSrc2, keypoints2);

	//显示两幅图的特征点
	Mat imgKeyPoints1;
	Mat imgKeyPoints2;

	drawKeypoints(imgSrc1, keypoints1, imgKeyPoints1, Scalar::all(-1),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(imgSrc2, keypoints2, imgKeyPoints2, Scalar::all(-1),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("图片1特征点", imgKeyPoints1);
	imshow("图片2特征点", imgKeyPoints2);

	//计算SURF的描述子
	SurfDescriptorExtractor extractor;

	Mat descriptors1, descriptors2;

	extractor.compute(imgSrc1, keypoints1, descriptors1);
	extractor.compute(imgSrc2, keypoints2, descriptors2);

	//使用FLANN匹配描述子
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;

	matcher.match(descriptors1, descriptors2, matches);

	//筛选出好的匹配对
	double maxDist = 0; double minDist = 100;

	//计算特征点之间的欧氏距离
	for (int i = 0; i < descriptors1.rows; i++)
	{
		double dist = matches[i].distance;

		if (dist < minDist)
		{
			minDist = dist;
		}

		if (dist > maxDist)
		{
			maxDist = dist;
		}
	}

	//欧氏距离小于等于2*minDist或0.02的匹配对保留

	std::vector< DMatch > goodMatches;

	for (int i = 0; i < descriptors1.rows; i++)
	{
		if (matches[i].distance <= max(2 * minDist, 0.02))
		{
			goodMatches.push_back(matches[i]);
		}
	}

	//显示经过筛选后的匹配情况,绿色为匹配对,红色为未匹配的单点
	Mat imgGoodMatches;

	drawMatches(imgSrc1, keypoints1, imgSrc2, keypoints2,
		goodMatches, imgGoodMatches, CV_RGB(0, 255, 0), CV_RGB(255, 0, 0),
		vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("Flann+Good Matches", imgGoodMatches);

	//RANSAC匹配

	int ptCount = (int)goodMatches.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);

	//KeyPoint转换为Mat
	Point2f pt;

	for (int i = 0; i<ptCount; ++i)
	{
		pt = keypoints1[goodMatches[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = keypoints2[goodMatches[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}

	// 用RANSAC方法计算F
	vector<uchar> RANSACStatus;       // 这个变量用于存储RANSAC后每个点的状态
	findFundamentalMat(p1, p2, RANSACStatus, FM_RANSAC);

	// 计算野点个数
	int OutlierCount = 0;

	for (int i = 0; i<ptCount; ++i)
	{
		if (RANSACStatus[i] == 0)    // 状态为0表示野点
		{
			OutlierCount++;
		}
	}

	int InlierCount = ptCount - OutlierCount;   // 计算内点

												// 这三个变量用于保存内点和匹配关系
	vector<Point2f> pointsInlier1;
	vector<Point2f> pointsInlier2;
	vector<DMatch> inlierMatches;

	pointsInlier1.resize(InlierCount);
	pointsInlier2.resize(InlierCount);
	inlierMatches.resize(InlierCount);

	InlierCount = 0;

	for (int i = 0; i<ptCount; ++i)
	{
		if (RANSACStatus[i] != 0)
		{
			pointsInlier1[InlierCount].x = p1.at<float>(i, 0);
			pointsInlier1[InlierCount].y = p1.at<float>(i, 1);
			pointsInlier2[InlierCount].x = p2.at<float>(i, 0);
			pointsInlier2[InlierCount].y = p2.at<float>(i, 1);

			inlierMatches[InlierCount].queryIdx = InlierCount;
			inlierMatches[InlierCount].trainIdx = InlierCount;

			InlierCount++;
		}
	}

	// 把内点转换为drawMatches可以使用的格式
	vector<KeyPoint> keypointsInlier1(InlierCount);
	vector<KeyPoint> keypointsInlier2(InlierCount);
	KeyPoint::convert(pointsInlier1, keypointsInlier1);
	KeyPoint::convert(pointsInlier2, keypointsInlier2);

	// 显示计算F过后的内点匹配（去除野点）
	Mat imgInlier;
	drawMatches(imgSrc1, keypointsInlier1, imgSrc2, keypointsInlier2,
		inlierMatches, imgInlier, Scalar::all(-1), CV_RGB(255, 0, 0),
		vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("RANSAC", imgInlier);

	waitKey(0);

	return 0;
}
