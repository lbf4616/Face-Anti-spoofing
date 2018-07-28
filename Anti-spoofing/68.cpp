#include <dlib/opencv.h>  
#include <opencv2/opencv.hpp>  
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>  
#include <dlib/gui_widgets.h>  
#include <vector>
#include <stdlib.h>
#include <opencv2/core/mat.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <sstream>
 
using namespace dlib;  
using namespace std;  
using namespace Eigen;

std::vector<double> rfft(std::vector<double> x);
std::vector<double> detrend(std::vector<double> x);
double max(std::vector<double> x);
double sum(std::vector<double> x);
std::vector<int> LBP(cv::Mat face, int P, int R);
 
double dur;
clock_t start, end, end1, end2, end3, end4, end5;


int main()
{	
	// Load face detection and pose estimation models.  
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	std::vector<double> B, G, R;
	// Grab and process frames until the main window is closed by the user.  
	for (int i = 1; i < 36; i++)
	{	
		for (int j = 1; j < 3; j++)
		{
			clock_t start, end;
			start = clock();
			string J = to_string(j);
			string VideoNum;
			string I = to_string(i);
			if (i < 10)
			{
				VideoNum = "0" + I + "_02_0" + J + ".mp4";
			}
			else 
			{
				VideoNum = I + "_02_0" + J + ".mp4";
			}
			//string dir = "D:\\Anti-spoofing\\databases for anti-spoofing\\3DMAD-spoofing\\videos\\03\\"+VideoNum;
			string dir = "D:\\Anti-spoofing\\databases for anti-spoofing\\MSU\\converted videos\\realvideo\\" + VideoNum;
			std::vector<double> B, G, R;
			cv::VideoCapture cap(dir); //capture video
			int nFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);//Total frames
			int times = 0; //Every three frames extract LBP features
			int t = 0; //Only extract HRfeature once
			int x01, x15, x03, x13, x05, y05, y03, y29, x09, x07, x10, x06, x11, x04, x12, y07, y06, y04;
			int B1 = 0, G1 = 0, R1 = 0;
			float Tpix1, Tpix2, Tpix3, Tpix4, Tpix5;
			std::vector<full_object_detection> shapes;
			for (int j = 0; j < nFrame; j++)
			//while (!win.is_closed())
			{
				cv::Mat temp;
				cap >> temp;
				t += 1;
				//temp = cv::imread("1.jpg", 1);
				// when using camera
				/*if (!cap.read(temp))
				{
					break;
				}*/  				
				if (times == 0)
				{
					cv_image<bgr_pixel> cimg(temp);
					// Detect faces   
					std::vector<rectangle> faces = detector(cimg);
					// Find the pose of each face.  
					for (unsigned long i = 0; i < faces.size(); ++i)
						shapes.push_back(pose_model(cimg, faces[i]));
					times++;
				}			
				if (!shapes.empty())
				{					
					/*circle(temp, cvPoint(x01, shapes[0].part(1).y()), 3, cv::Scalar(0, 0, 255), -1);
					circle(temp, cvPoint(x03, shapes[0].part(3).y()), 3, cv::Scalar(0, 0, 255), -1);
					circle(temp, cvPoint(x05, shapes[0].part(5).y()), 3, cv::Scalar(0, 0, 255), -1);
					circle(temp, cvPoint(shapes[0].part(8).x(), y08), 3, cv::Scalar(0, 0, 255), -1);
					circle(temp, cvPoint(x11, shapes[0].part(11).y()), 3, cv::Scalar(0, 0, 255), -1);
					circle(temp, cvPoint(x13, shapes[0].part(13).y()), 3, cv::Scalar(0, 0, 255), -1);
					circle(temp, cvPoint(x15, shapes[0].part(15).y()), 3, cv::Scalar(0, 0, 255), -1);
					circle(temp, cvPoint(x30, shapes[0].part(30).y()), 3, cv::Scalar(0, 0, 255), -1);*/
					
					//HRfeature extracting, Get ROI
					x01 = shapes[0].part(1).x() + 5;
					x15 = shapes[0].part(15).x() - 5;
					x03 = shapes[0].part(3).x() + 5;
					x13 = shapes[0].part(13).x() - 5;
					x05 = shapes[0].part(5).x() + 5;
					y05 = shapes[0].part(5).y();
					y03 = shapes[0].part(3).y();
					y29 = shapes[0].part(1).y();
					x09 = shapes[0].part(9).x() + 5;
					x07 = shapes[0].part(7).x() - 5;
					x10 = shapes[0].part(10).x() - 5;
					x06 = shapes[0].part(6).x() + 5;
					x11 = shapes[0].part(11).x() - 5;
					x04 = shapes[0].part(4).x() + 5;
					x12 = shapes[0].part(12).x() - 5;
					y07 = shapes[0].part(7).y();
					y06 = shapes[0].part(6).y();
					y04 = shapes[0].part(4).y();

					B1 = 0, G1 = 0, R1 = 0;
					int ROI1L = x09 - x07;     //set ROI1
					int ROI1H = y07 - y06;
					Tpix1 = ROI1L * ROI1H;
					int ROI2L = x10 - x06;     //2
					int ROI2H = y06 - y05;
					Tpix2 = ROI2L * ROI2H;
					int ROI3L = x11 - x05;     //3
					int ROI3H = y05 - y04;
					Tpix3 = ROI3L * ROI3H;
					int ROI4L = x12 - x04;     //4
					int ROI4H = y04 - y03;
					Tpix4 = ROI4L * ROI4H;
					int ROI5L = x13 - x03;     //5
					int ROI5H = y03 - y29;
					Tpix5 = ROI5L * ROI5H;

					// Calculate the ROI1 Blue
					for (int j = y06; j < y07 + 1; j++)
					{
						for (int i = x07; i < x09 + 1; i++)
						{
							B1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 0];
						}
					}
					// Calculate the ROI2 Blue
					for (int j = y05; j < y06 + 1; j++)
					{
						for (int i = x06; i < x10 + 1; i++)
						{
							B1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 0];
						}
					}
					// Calculate the ROI3 Blue
					for (int j = y04; j < y05 + 1; j++)
					{
						for (int i = x05; i < x11 + 1; i++)
						{
							B1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 0];
						}
					}
					// Calculate the ROI4 Blue
					for (int j = y03; j < y04 + 1; j++)
					{
						for (int i = x04; i < x12 + 1; i++)
						{
							B1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 0];
						}
					}
					// Calculate the ROI5 Blue
					for (int j = y29; j < y03 + 1; j++)
					{
						for (int i = x03; i < x13 + 1; i++)
						{
							B1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 0];
						}
					}


					// Calculate the ROI1 Green
					for (int j = y06; j < y07 + 1; j++)
					{
						for (int i = x07; i < x09 + 1; i++)
						{
							G1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 1];
						}
					}
					// Calculate the ROI2 Green
					for (int j = y05; j < y06 + 1; j++)
					{
						for (int i = x06; i < x10 + 1; i++)
						{
							G1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 1];
						}
					}
					// Calculate the ROI3 Green
					for (int j = y04; j < y05 + 1; j++)
					{
						for (int i = x05; i < x11 + 1; i++)
						{
							G1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 1];
						}
					}
					// Calculate the ROI4 Green
					for (int j = y03; j < y04 + 1; j++)
					{
						for (int i = x04; i < x12 + 1; i++)
						{
							G1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 1];
						}
					}
					// Calculate the ROI5 Green
					for (int j = y29; j < y03 + 1; j++)
					{
						for (int i = x03; i < x13 + 1; i++)
						{
							G1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 1];
						}
					}


					// Calculate the ROI1 Red
					for (int j = y06; j < y07 + 1; j++)
					{
						for (int i = x07; i < x09 + 1; i++)
						{
							R1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 2];
						}
					}
					// Calculate the ROI2 Red
					for (int j = y05; j < y06 + 1; j++)
					{
						for (int i = x06; i < x10 + 1; i++)
						{
							R1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 2];
						}
					}
					// Calculate the ROI3 Red
					for (int j = y04; j < y05 + 1; j++)
					{
						for (int i = x05; i < x11 + 1; i++)
						{
							R1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 2];
						}
					}
					// Calculate the ROI4 Red
					for (int j = y03; j < y04 + 1; j++)
					{
						for (int i = x04; i < x12 + 1; i++)
						{
							R1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 2];
						}
					}
					// Calculate the ROI5 Red
					for (int j = y29; j < y03 + 1; j++)
					{
						for (int i = x03; i < x13 + 1; i++)
						{
							R1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 2];
						}
					}
					float avgB = B1 / (Tpix1 + Tpix2 + Tpix3 + Tpix4 + Tpix5);  // All points get average
					float avgG = G1 / (Tpix1 + Tpix2 + Tpix3 + Tpix4 + Tpix5);
					float avgR = R1 / (Tpix1 + Tpix2 + Tpix3 + Tpix4 + Tpix5);

					B.push_back(avgB);
					G.push_back(avgG);
					R.push_back(avgR);			

					//Eyes inner corner coordinates;
					int x39 = shapes[0].part(39).x();
					int x42 = shapes[0].part(42).x();
					int y39 = shapes[0].part(39).y();
					int y42 = shapes[0].part(42).y();
					if (t % 3 == 0)
					{
						cv::Rect rect(x39 - 1.5 * (x42 - x39), y39 - 1.5 * (x42 - x39), 4 * (x42 - x39), 4 * (x42 - x39));
						cv::Mat image_cut = temp(rect);      //从img中按照rect进行切割
						//cv::Mat image_copy = image_cut.clone();
						//cv::imshow("456", image_cut);
						cv::Mat dst;
						cv::resize(image_cut, dst, cv::Size(128, 128), CV_INTER_LINEAR);
						//cv::imwrite("d://"+ I +".jpg", dst);
						std::vector<int> lbp1, lbp2, lbp3, lbp4;
						//cout << dst.cols << endl;
						lbp1 = LBP(dst, 8, 1);
						lbp2 = LBP(dst, 8, 2);
						lbp3 = LBP(dst, 8, 3);
						lbp4 = LBP(dst, 8, 4);
						ofstream ofs1;
						char filename[19] = "d://LBPfeature.csv";
						ofs1.open(filename, ostream::app);//以添加模式打开文件
						ofs1 << VideoNum;
						for (int i = 0; i < lbp1.size(); i++)
						{
							ofs1 << " ," << lbp1[i];
						}
						for (int i = 0; i < lbp2.size(); i++)
						{
							ofs1 << " ," << lbp2[i];
						}
						for (int i = 0; i < lbp3.size(); i++)
						{
							ofs1 << " ," << lbp3[i];
						}
						for (int i = 0; i < lbp4.size(); i++)
						{
							ofs1 << " ," << lbp4[i];
						}
						ofs1 << "," << 2 << endl; //写入数据
						ofs1.close(); //关闭文件
						ofs1.clear(); //清理
					}
				}
				/*win.clear_overlay();
				win.set_image(cimg);
				win.add_overlay(render_face_detections(shapes));*/
			}
			/*for (int i = 0; i < B.size(); i++)
			{
				cout << B[i] << ",";
			}
			cout << endl;

			for (int i = 0; i < G.size(); i++)
			{
				cout << G[i] << ",";
			}
			cout << endl;
			

			for (int i = 0; i < R.size(); i++)
			{
			cout << R[i] << ",";
			}
			cout << endl;*/
			
			std::vector<double> Yr, Yg, Yb;
			std::vector<double> Xr, Xg, Xb;
			Yb = detrend(B);
			Yg = detrend(G);
			Yr = detrend(R);		
			Xb = rfft(Yb);
			Xg = rfft(Yg);
			Xr = rfft(Yr);

			double Er, Eg, Eb, Tr, Tg, Tb;
			Er = max(Xr);
			Eg = max(Xg);
			Eb = max(Xb);
			Tr = Er / sum(Xr);
			Tg = Eg / sum(Xg);
			Tb = Eb / sum(Xb);

			ofstream ofs;
			char filename[16] = "d://data.csv";
			ofs.open(filename, ostream::app);//以添加模式打开文件
			ofs << VideoNum << " ," << Er << ", " << Eg << ", " << Eb << ", " << Tr << ", " << Tg << ", " << Tb << "," << 2 <<endl; //写入数据
			ofs.close(); //关闭文件
			ofs.clear(); //清理

			cout << VideoNum <<" "<< "done" << endl;		
			end = clock();		
			dur = (double)(end - start);
			printf("Use Time:%f\n", (dur / CLOCKS_PER_SEC));
		}
	}	
	std::system("pause");
}

std::vector<double> detrend(std::vector<double> x)
{
	start = clock();
	int lambda = 50;
	int BL = x.size();
	MatrixXd Color(1, BL);
	for (int i = 0; i < BL; i++)
	{
		Color(0, i) = x[i];
	}
	int L = Color.cols();
	int m = L - 2, n = L;
	SparseMatrix<double> D2(m, n);
	D2.reserve(VectorXi::Constant(n, 3));
	for (int i = 0; i < m; ++i)
	{
		D2.insert(i, i) = 1;
		D2.insert(i, i + 1) = -2;
		D2.insert(i, i + 2) = 1;
	}
	D2.makeCompressed();
	SparseMatrix<double>  I(L, L);
	//Sparse eye I
	for (int i = 0; i < L; ++i)
	{
		I.insert(i, i) = 1;
	}
	SparseMatrix<double> ColorD1(L, L);
	ColorD1 = (I + (pow(lambda, 2) * (D2.adjoint()*D2)));
	//Sparse Matrix to Matrix
	MatrixXd ColorD2 = ColorD1;
	MatrixXd I2 = I;
	ColorD2 = (I2 - ColorD2.inverse())* (Color.adjoint());
	MatrixXd ColorD3 = ColorD2.adjoint();

	//Moving average filter
	int N = 5;
	MatrixXd sa(1, L);
	sa.setZero(1, L);
	std::vector<double> sla;
	double a;

	// Matrix to Vector, while set N-1 zeros at the begining of the vector
	for (int i = 0; i < N - 1; i++)
	{
		sla.push_back(0);
	}
	for (int i = 0; i < L; i++)
	{
		sla.push_back(ColorD3(0, i));
	}

	//Filter
	for (int i = 1; i <= L; i++)
	{
		for (int j = 0; j < N; j++)
		{
			a += sla[i + j - 1] / N;
		}
		sa(0, i - 1) = a;
		a = 0;
	}

	//FIR Hd
	int n1 = 127, n2; //n orders of the filter
	double fln = 0.7, fhn = 4; //下届，上界
	double s, pi, wc1, wc2, delay, h[127], fs = 30;
	pi = 4.0*atan(1.0);//pi=PI;
	n2 = n1 / 2;
	delay = n1 / 2.0;
	wc1 = 2.0*pi*fln;
	wc2 = 2.0*pi*fhn;
	for (int i = 0; i <= n2; i++)
	{
		s = i - delay;
		h[i] = ((sin(wc2*s / fs) - sin(wc1*s / fs)) / (pi*s))* (0.54 - 0.46*cos(2 * i*pi / (n1 - 1)));//Bandpass with Hamming window	
		h[n1 - i] = h[i];
		//h[n1 / 2] = (wc2 - wc1) / pi;
	}
	//for (int i = 0; i <= n1; i++)
	//{
	//	cout << h[i] << endl;
	//}
	std::vector<double> Y;
	std::vector<double> X;
	double y;
	for (int j = 0; j < L; j++)
	{
		for (int i = 0; i < n1; i++)
		{
			if (j >= i)
				y += h[i] * sa(0, j - i);
		}
		Y.push_back(y);
		//cout << y << endl;
	}
	return Y;
}

//快速傅里叶变换
std::vector<double> rfft(std::vector<double> x)
{
	//std::vector<double> x = {1,2,3,4,5};
	int xz1 = x.size();
	int xz = x.size(); //必须是2的整数次幂
	int n;
	char s[20];
	itoa(xz, s, 2);//转为二进制
	string y = s;  //转为字符串
	int L = y.length();
	for (int i = 1; i < L; i++) //除第一位以外有1的
	{
		if (y[i] == '1')
		{
			n = pow(2, L);  //Next higher power of 2
			for (int j = xz; xz < n; xz++)//扩展后的x的后面用0补足
			{
				x.push_back(0);
			}
			break;
		}
	}

	int m, k, n1, n2, n4,i1, i2, i3, i4;
	double a, e, cc, ss, xt, t1, t2;
	
	for (int j = 1, i = 1; i < 16; i++)
	{
		m = i;
		j = 2 * j;
		if (j == n)break;
	}
	n1 = n - 1;
	for (int j = 0, i = 0; i < n1; i++)
	{
		if (i < j)
		{
			xt = x[j];
			x[j] = x[i];
			x[i] = xt;
		}
		k = n / 2;
		while (k < (j + 1))
		{
			j = j - k;
			k = k / 2;
		}
		j = j + k;
	}
	for (int i = 0; i < n; i += 2)
	{
		xt = x[i];
		x[i] = xt + x[i + 1];
		x[i + 1] = xt - x[i + 1];
	}
	n2 = 1;
	for (int k = 2; k <= m; k++)
	{
		n4 = n2;
		n2 = 2 * n4;
		n1 = 2 * n2;
		e = 6.2831850718 / n1;
		for (int i = 0; i < n; i += n1)
		{
			xt = x[i];
			x[i] = xt + x[i + n2];
			x[i + n2] = xt - x[i + n2];
			x[i + n2 + n4] = -x[i + n2 + n4];
			a = e;
			for (int j = 1; j <= (n4 - 1); j++)
			{
				i1 = i + j;
				i2 = i - j + n2;
				i3 = i + j + n2;
				i4 = i - j + n1;
				cc = cos(a);
				ss = sin(a);
				a = a + e;
				t1 = cc*x[i3] + ss*x[i4];
				t2 = ss*x[i3] - cc*x[i4];
				x[i4] = x[i2] - t2;
				x[i3] = -x[i2] - t2;
				x[i2] = x[i1] - t1;
				x[i1] = x[i1] + t1;
			}
		}
	}
	std::vector<double> p;
	p.push_back(2 * abs(x[0])/xz1);
	for (int i = 1; i < n / 2; i++)
	{
		p.push_back(2 * sqrt(pow(x[i], 2) + pow(x[n - i], 2))/xz1);
		//cout << p[i] << endl;
	}
	p.push_back(2 * abs(x[n / 2])/xz1);
	return p;
}

//Find max frequency
double max(std::vector<double> x)
{
	int Xsize = x.size();
	double max = 0;
	for (int i = 12; i < 70; i++)//频率在13~70之间
	{
		if (x[i] > max)
		{
			max = x[i];
		}
	}
	return max;
}

//Calculate the sum of frequency
double sum(std::vector<double> x)
{
	int Xsize = x.size();
	double sum = 0;
	for (int i = 12; i < 70; i++)
	{
			sum += x[i];
	}
	return sum;
}

//LBP with uniform pattern 59 dimensional vector
std::vector<int> LBP(cv::Mat face, int P, int R)
{
	std::vector<int> histogram;
	std::vector<int> feature;
	// Split the image into different channels
	int LBP[59] = { 0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255 };	

	//Extract LBP features	Green
	for (int i = R; i < face.cols - R; i++)
	{	

		for (int j = R; j < face.rows - R; j++)
		{
			int temp[] = { face.at<cv::Vec3b>(i - R, j - R)[0], face.at<cv::Vec3b>(i, j - R)[0], face.at<cv::Vec3b>(i + R, j - R)[0], face.at<cv::Vec3b>(i + R, j)[0], +face.at<cv::Vec3b>(i + R, j + R)[0], face.at<cv::Vec3b>(i, j + R)[0], face.at<cv::Vec3b>(i - R, j + R)[0], face.at<cv::Vec3b>(i - R, j)[0] };
			int value = 0;
			for (int k = 0; k < 8; k++)
			{
				if (face.at<cv::Vec3b>(i, j)[0] > temp[k])
				{
					value += pow(double(2), k);				
				}
			}	
			feature.push_back(value);			
		}		
	}

	//Count the histogram
	for (int i = 0; i < sizeof(LBP)/4; i++)
	{
		int x = count(feature.begin(), feature.end(), LBP[i]);
		histogram.push_back(x);
	}

	//Extract LBP features	Blue
	for (int i = R; i < face.cols - R; i++)
	{

		for (int j = R; j < face.rows - R; j++)
		{
			int temp[] = { face.at<cv::Vec3b>(i - R, j - R)[1], face.at<cv::Vec3b>(i, j - R)[1], face.at<cv::Vec3b>(i + R, j - R)[1], face.at<cv::Vec3b>(i + R, j)[1], +face.at<cv::Vec3b>(i + R, j + R)[1], face.at<cv::Vec3b>(i, j + R)[1], face.at<cv::Vec3b>(i - R, j + R)[1], face.at<cv::Vec3b>(i - R, j)[1] };
			int value = 0;
			for (int k = 0; k < 8; k++)
			{
				if (face.at<cv::Vec3b>(i, j)[1] > temp[k])
				{
					value += pow(double(2), k);
				}
			}
			feature.push_back(value);
		}
	}

	//Count the histogram
	for (int i = 0; i < sizeof(LBP) / 4; i++)
	{
		int x = count(feature.begin(), feature.end(), LBP[i]);
		histogram.push_back(x);
	}

	//Extract LBP features	Red
	for (int i = R; i < face.cols - R; i++)
	{

		for (int j = R; j < face.rows - R; j++)
		{
			int temp[] = { face.at<cv::Vec3b>(i - R, j - R)[2], face.at<cv::Vec3b>(i, j - R)[2], face.at<cv::Vec3b>(i + R, j - R)[2], face.at<cv::Vec3b>(i + R, j)[2], +face.at<cv::Vec3b>(i + R, j + R)[2], face.at<cv::Vec3b>(i, j + R)[2], face.at<cv::Vec3b>(i - R, j + R)[2], face.at<cv::Vec3b>(i - R, j)[2] };
			int value = 0;
			for (int k = 0; k < 8; k++)
			{
				if (face.at<cv::Vec3b>(i, j)[2] > temp[k])
				{
					value += pow(double(2), k);
				}
			}
			feature.push_back(value);
		}
	}

	//Count the histogram
	for (int i = 0; i < sizeof(LBP) / 4; i++)
	{
		int x = count(feature.begin(), feature.end(), LBP[i]);
		histogram.push_back(x);
	}
	return histogram;
}