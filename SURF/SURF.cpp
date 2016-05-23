#include <cstdio>
#include <iostream>
#include <opencv2/nonfree/nonfree.hpp>  
#include<opencv2/opencv.hpp>

using namespace cv;

int main()
{
	//��ȡͼ��
	Mat imgSrc1 = imread("asong1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgSrc2 = imread("asong2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	//ʹ��SURF���������
	int minHessian = 4000;

	SurfFeatureDetector detector(minHessian);

	std::vector<KeyPoint> keypoints1, keypoints2;

	detector.detect(imgSrc1, keypoints1);
	detector.detect(imgSrc2, keypoints2);

	//��ʾ����ͼ��������
	Mat imgKeyPoints1;
	Mat imgKeyPoints2;

	drawKeypoints(imgSrc1, keypoints1, imgKeyPoints1, Scalar::all(-1),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(imgSrc2, keypoints2, imgKeyPoints2, Scalar::all(-1),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("ͼƬ1������", imgKeyPoints1);
	imshow("ͼƬ2������", imgKeyPoints2);

	//����SURF��������
	SurfDescriptorExtractor extractor;

	Mat descriptors1, descriptors2;

	extractor.compute(imgSrc1, keypoints1, descriptors1);
	extractor.compute(imgSrc2, keypoints2, descriptors2);

	//ʹ��FLANNƥ��������
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;

	matcher.match(descriptors1, descriptors2, matches);

	//ɸѡ���õ�ƥ���
	double maxDist = 0; double minDist = 100;

	//����������֮���ŷ�Ͼ���
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

	//ŷ�Ͼ���С�ڵ���2*minDist��0.02��ƥ��Ա���

	std::vector< DMatch > goodMatches;

	for (int i = 0; i < descriptors1.rows; i++)
	{
		if (matches[i].distance <= max(2 * minDist, 0.02))
		{
			goodMatches.push_back(matches[i]);
		}
	}

	//��ʾ����ɸѡ���ƥ�����,��ɫΪƥ���,��ɫΪδƥ��ĵ���
	Mat imgGoodMatches;

	drawMatches(imgSrc1, keypoints1, imgSrc2, keypoints2,
		goodMatches, imgGoodMatches, CV_RGB(0, 255, 0), CV_RGB(255, 0, 0),
		vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("Flann+Good Matches", imgGoodMatches);

	//RANSACƥ��

	int ptCount = (int)goodMatches.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);

	//KeyPointת��ΪMat
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

	// ��RANSAC��������F
	vector<uchar> RANSACStatus;       // ����������ڴ洢RANSAC��ÿ�����״̬
	findFundamentalMat(p1, p2, RANSACStatus, FM_RANSAC);

	// ����Ұ�����
	int OutlierCount = 0;

	for (int i = 0; i<ptCount; ++i)
	{
		if (RANSACStatus[i] == 0)    // ״̬Ϊ0��ʾҰ��
		{
			OutlierCount++;
		}
	}

	int InlierCount = ptCount - OutlierCount;   // �����ڵ�

												// �������������ڱ����ڵ��ƥ���ϵ
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

	// ���ڵ�ת��ΪdrawMatches����ʹ�õĸ�ʽ
	vector<KeyPoint> keypointsInlier1(InlierCount);
	vector<KeyPoint> keypointsInlier2(InlierCount);
	KeyPoint::convert(pointsInlier1, keypointsInlier1);
	KeyPoint::convert(pointsInlier2, keypointsInlier2);

	// ��ʾ����F������ڵ�ƥ�䣨ȥ��Ұ�㣩
	Mat imgInlier;
	drawMatches(imgSrc1, keypointsInlier1, imgSrc2, keypointsInlier2,
		inlierMatches, imgInlier, Scalar::all(-1), CV_RGB(255, 0, 0),
		vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("RANSAC", imgInlier);

	waitKey(0);

	return 0;
}
