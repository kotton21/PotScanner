/*
 * test.cpp
 *
 *  Created on: Jan 2, 2017
 *      Author: karl
 */


#include <cv.h>
#include <highgui.h>

using namespace cv;

//#include <stdio>
//#include <dirent.h>
#include <cstdlib>
#include <cmath>
#include <ctgmath>
#include <math.h>       /* sin */
#define PI 3.14159265
using std::cout;
using std::endl;



//points in images (x, y)
class ImagePoint : public Point {
public:
	ImagePoint(int x, int y) {
		this->x = x;
		this->y = y;
	}
};

//points in theta, phi, z.
class AnglePoint {
public:
	AnglePoint(float theta, float phi, float z)
		: theta(theta), phi(phi), z(z)
		{ }
	float theta;
	float phi;
	float z;
};

class LinearEqn {
public:
	float m;
	float b;

	LinearEqn(float m, float b)
			: m(m), b(b) { } // simple constructor
	LinearEqn(const LinearEqn &eqn)
		: m(eqn.m), b(eqn.b) { } // copy constructor
	~LinearEqn() { } // destructor

	float y(float x) {
		return m*x + b;
	}

	Point2f Intersection(LinearEqn eqn) {
		float x = (eqn.b - this->b) / (this->m - eqn.m);
		float y = this->m*x + this->b;
		return Point2f(x,y);
	}
};

// this accounts to creating a type in order to handle a single variable
// wouldn't a vector of vectors work just fine?
//class AnglePointMap: Mat {
//public:
//	AnglePointMap(int height, int numAngles) : Mat(height, numAngles, CV_32FC3) {}
//	void set(int y, int x, AnglePoint pt) {
//
//	}
//	AnglePoint get(int y, int x) {
//
//	}
//};

using AnglePointMap = vector<vector<AnglePoint>>;

//a 2d map of any type. must provide own functionality for resizing
template<typename T>class VectorMap {
public:
	vector<vector<T>> vmap;
	int height;
	int width;

	VectorMap(int height, int width, T defaultFill) : height(height), width(width) {
		vmap.reserve(height);
		for (int h=0; h<height; ++h){
			vmap.push_back( vector<T>(width, defaultFill));
		}
	}
//	VectorMap(&VectorMap obj) : VectorMap(obj.height, obj.width, ) { }

	void set(const int y, const int x, T& pt) {
		vmap.at(y).at(x) = pt;
	}
	T get(const int y, const int x) {
		return vmap.at(y).at(x);
	}
	void doFunc(void (*func)(VectorMap&, VectorMap&, int, int), VectorMap& toMap) { //B //
		for (int h=0; h < height; ++h) {
			for (int w=0; w < width; ++w) {
				func(*this,toMap,h,w); //
			}
		}
	}
};

void funcDoIntersections(VectorMap<LinearEqn>& from, VectorMap<Point2f>& to, int h, int w) { // to
	LinearEqn curr = from.get(h,w);
	LinearEqn next = from.get(h,(w+1)%from.width);

	Point2f intersection = curr.Intersection(next);
	to.set(h,w,intersection);
}


void dispImage(Mat image) {
	namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
	imshow( "Display Image", image );
	waitKey(0);
}



void dispEdge(const vector<ImagePoint> edges, Mat img, const int centerCol = 0) {
	cout << "edges: " << edges.size() << endl;
	for(auto vert: edges) {
		circle(img, vert, 3, Scalar(0,0,255), 2);
	}
	if (centerCol > 0) {
		for (int r = 0; r<img.rows; ++r) {
			img.at<Vec3b>(r,centerCol) = Vec3b(0,0,255);
		}
	}
	dispImage(img);
}



vector<ImagePoint> detectEdges(const Mat hsv, const int NUMVERTSTEPS, const int CENTERCOL) {
	//Mat detectEdges(Mat hsv) {
	vector<ImagePoint> edges;
	edges.reserve(NUMVERTSTEPS);
	int pixelstep = hsv.rows / NUMVERTSTEPS;

//	int edges[hsv.rows/10] = {};
	//Mat edges (1, hsv.rows/10, CV_16UC2, 0);

	int r = 0, i = 0;
	while (r < hsv.rows) {
		bool found = false;
		for (int c=0; c < hsv.cols; c++) {
			//hue from 205-265, saturation > 50%, value > 30% ?
			Vec3b pt = hsv.at<Vec3b>(r,c);

			//why the hell is the hue value so small? It never goes above 17.
			if (int(pt.val[0]) > 100 && int(pt.val[0]) < 135 &&
					int(pt.val[1]) > 125 &&
					int(pt.val[2]) > 100) {
				//save point
				ImagePoint vert (c,r);
				edges.push_back(vert);
				//edges.at<Point>(i) = vert;

				//skip to next row
				found = true;
				break;
			}
		}
		if (!found) { edges.push_back(ImagePoint(CENTERCOL,r)); }

		i = i + 1;
		r = r + pixelstep;
	}
	return edges;
}


//Certainly not optimized for speed lolz
//also, this is only valid for orthographic views.
vector<Point3f> getCartesion(vector<Point> points, int theta, int centerCol) {
	vector<Point3f> polar;  // r, theta, z representation
	for (auto point: points) {
		Point3f ppt (abs(centerCol - point.x), theta, point.y);
		polar.push_back(ppt);
	}

	vector<Point3f> cartesian;  // x, y, z representation
	for (auto point: polar) {
		Point3f ppt (point.x * cos(point.y * PI / 180.),
				point.x * sin(point.y * PI / 180.),
				point.z);
		polar.push_back(ppt);
	}

	return cartesian;
}



vector<AnglePoint> getThetas(const vector<ImagePoint> edges, const float focalDist, const float phi) {
	vector<AnglePoint> thetas;
	thetas.reserve(edges.size());
	for (auto edge: edges) {
		float theta = atan(edge.x/focalDist);
		thetas.push_back(AnglePoint(theta, phi, edge.y));
	}
	return thetas;
}



//class OrthoFrameGetter {
//public:
//
//};

//vector<string> filenames(const char* dir) {
//	DIR *dirp;
//	int len = strlen(dir);
//	dirp = opendir(".");
//	while ((dp = readdir(dirp)) != NULL)
//		   if (dp->d_namlen == len && !strcmp(dp->d_name, dir)) {
//				   (void)closedir(dirp);
//				   return FOUND;
//		   }
//	(void)closedir(dirp);
//	return NOT_FOUND;
//}


int main( int argc, char** argv ) {

	vector<string> filenames = {"0.jpg"}; //, "30.jpg", "60.jpg",
//			"90.jpg", "120.jpg", "150.jpg",
//			"240.jpg", "270.jpg", "300.jpg", "330.jpg"};

	Mat image;
	//image = imread( argv[1], 1 );
	Rect roi (1000,1000,500,500); //user input
	int CENTERCOL = 417; //user input or determined by machine architecture
	int NUMANGLES = filenames.size();
	int NUMVERTSTEPS = 10;
	float FOCALDIST = 200;

	AnglePointMap apMap;
	apMap.reserve(NUMANGLES);

	float phi = 0;
	for (unsigned int i = 0; i < filenames.size(); ++i) {
		// Region of Interest
		image = imread( "pics/"+filenames[i], 1 );
		Mat image_roi(image, Rect(2500, 500, 800, 600)); // x0, y0, w, h
		Mat hsv;
		cvtColor(image_roi, hsv, CV_BGR2HSV);

		//Mat edges = detectEdges(hsv);
		vector<ImagePoint> edges = detectEdges(hsv, NUMVERTSTEPS, CENTERCOL);

		vector<AnglePoint> anglePoints = getThetas(edges, FOCALDIST, phi);
		phi = phi + 30.0;

		apMap.push_back(anglePoints);

		//dispEdge(edges, image_roi, CENTERCOL);
	}

	//now have a 2d vector of AnglePoints. Build the map.
	//VectorMap considers the texture as rolled off. Need to transpose the AnglePoints.
	//This is starting to look an awfull lot like i'm just writing a linear algebra library.
	VectorMap<LinearEqn> eqns (NUMVERTSTEPS, NUMANGLES, LinearEqn(0,0));
	for (int h=0; h<NUMVERTSTEPS; ++h) {
		for (int w=0; w<NUMANGLES; ++w) {
			AnglePoint pt = apMap.at(w).at(h);
			float m = tan(pt.theta);
			float b = m*FOCALDIST;
			LinearEqn thiseqn (m,b);
			eqns.set(h,w, thiseqn);
		}
	}

	//Now have a map of linear equations.. no obvious float range problems so far.
	//Build Intersection Points? Can I do this in the existing map?
	//Give LinearEqn knowledge of its adjacent item?
	VectorMap<Point2f> intersections (NUMVERTSTEPS, NUMANGLES, Point2f(0,0));;
	eqns.doFunc(&funcDoIntersections, intersections); //;, intersections


//	dispEdge(edges, image_roi, centerCol);
//	Mat clipped;
//	inRange(hsv, Scalar(100, 100, 100), Scalar(135,255,255), clipped);
//
//	dispImage(clipped);

	return 0;
}

