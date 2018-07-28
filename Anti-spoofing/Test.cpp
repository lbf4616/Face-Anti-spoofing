#include <fstream>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <iostream>
using namespace std;
using namespace cv;
using namespace cv::ml;

Mat read_train_feature(const string fileName, string Out);
Mat read_test_feature(const string fileName, string Out);
Mat read_train_label(const string fileName, string Out);
Mat read_test_label(const string fileName, string Out);

string Data = "d://LBPfeature.csv";

int main()
{
	//读取训练数据
	Mat trainData;
	Mat labels;
	string Out;
	int totalT = 0;
	//for (int i = 1; i < 18; i++)
	//{
	//	if (i < 10)
	//	{
	//		Out = "0" + to_string(i);
	//	}
	//	else 
	//	{
	//		Out = to_string(i);
	//	}	
	//Out = "01";
	trainData = read_train_feature(Data, Out);
	labels = read_train_label(Data, Out);
	//训练参数
	Ptr<SVM> svm = SVM::create();
	/*svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setGamma(0.01);
	svm->setC(10.0);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON));*/
	//训练分类器
	svm->trainAuto(trainData, ROW_SAMPLE, labels);
	//保存训练器
	svm->save("d://LBP.txt");
	//cout << "save as /mnist_dataset/mnist_svm.xml" << endl;
	//下载分类器
	//cout << "开始导入SVM文件...\n";
	//Ptr<SVM> svm1 = StatModel::load<SVM>("mnist_dataset/mnist_svm.xml");
	//cout << "成功导入SVM文件...\n";
	//读取测试数据
	Mat testData;
	Mat tLabel;
	testData = read_test_feature(Data, Out);
	tLabel = read_test_label(Data, Out);

	float count = 0;
	for (int i = 0; i < testData.rows; i++)
	{
		Mat sample = testData.row(i);
		float res = 0.0;
		res = svm->predict(sample);
		//cout << res << endl;
		res = abs(res - tLabel.at<unsigned int>(i, 0)) <= FLT_EPSILON ? 1.f : 0.f;
		count += res;
	}
	totalT += count;
	cout << "准确率为：" << count / testData.rows * 100 << "%\n";

	cout << totalT / 1.7 << "% " << endl;
	std::system("pause");
	return 0;
}

Mat read_train_feature(const string fileName, string Out)
{
	ifstream ofs;
	ofs.open(fileName, ios::in);
	string lineStr;
	int i = 0;
	string Num, Num2;
	Mat DataMat;
	//DataMat = Mat::zeros(240, 6, CV_32FC1);
	DataMat = Mat::zeros(19083, 708, CV_32FC1);
	while (getline(ofs, lineStr))
	{
		string cut = lineStr.substr(14);
		Num = lineStr.substr(0, 2);
		Num2 = lineStr.substr(3, 2);
		if (Num != "01" && Num != "02" && Num != "03")
		{
			for (int j = 0; j < 708; j++) //708 or 6
			{
				string E1 = cut.substr(0, cut.find_first_of(","));
				cut = cut.substr(cut.find_first_of(",") + 1);
				DataMat.at<float>(i, j) = (atof(E1.c_str()));			
			}
			i++;			
		}			
	}
	ofs.close(); //关闭文件
	ofs.clear(); //清理
	return DataMat;
}

Mat read_train_label(const string fileName, string Out)
{
	ifstream ofs;
	ofs.open(fileName, ios::in);
	string lineStr;
	int i = 0;
	Mat DataMat;
	string Num, Num2;
	DataMat = Mat::zeros(19083, 1, CV_32SC1);
	while (getline(ofs, lineStr))
	{
		Num = lineStr.substr(0, 2);
		Num2 = lineStr.substr(3, 2);
		if (Num != "01" && Num != "02" && Num != "03")
		{
			string E1 = lineStr.substr(lineStr.find_last_of(",") + 1);
			DataMat.at<int>(i, 0) = (atof(E1.c_str()));
			i++;
		}	
	}
	ofs.close(); //关闭文件
	ofs.clear(); //清理
	return DataMat;
}
Mat read_test_feature(const string fileName, string Out)
{
	ifstream ofs;
	ofs.open(fileName, ios::in);
	string lineStr;
	int i = 0;
	Mat DataMat;
	DataMat = Mat::zeros(1784, 708, CV_32FC1);
	string Num, Num2;
	while (getline(ofs, lineStr))
	{
		string cut = lineStr.substr(14);
		Num = lineStr.substr(0, 2);
		Num2 = lineStr.substr(3, 2);
		if (Num == "01" || Num == "02" || Num == "03")//&& Num2 != "02"
		{
			for (int j = 0; j < 708; j++) //708 or 6
			{
				string E1 = cut.substr(0, cut.find_first_of(","));
				cut = cut.substr(cut.find_first_of(",") + 1);
				DataMat.at<float>(i, j) = (atof(E1.c_str()));
			}
			i++;
		}	
	}
	ofs.close(); //关闭文件
	ofs.clear(); //清理
	return DataMat;
}
Mat read_test_label(const string fileName, string Out)
{
	ifstream ofs;
	ofs.open(fileName, ios::in);
	string lineStr;
	int i = 0;
	Mat DataMat;
	string Num, Num2;
	DataMat = Mat::zeros(1784, 1, CV_32SC1);
	while (getline(ofs, lineStr))
	{
		Num = lineStr.substr(0, 2);
		Num2 = lineStr.substr(3, 2);
		if (Num == "01" || Num == "02" || Num == "03") // && Num2 != "02"
		{
			string E1 = lineStr.substr(lineStr.find_last_of(",") + 1);
			DataMat.at<int>(i, 0) = (atof(E1.c_str()));
			i++;
		}	
	}
	ofs.close(); //关闭文件
	ofs.clear(); //清理
	return DataMat;
}