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
	ImagePoint(const ImagePoint& obj) : ImagePoint(obj.x, obj.y) { }
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

//Converted to 3d
class LinearEqn {
public:
	float m;
	float b;
	float z;

	LinearEqn(float m, float b, float z)
			: m(m), b(b), z(z) { } // simple constructor
	LinearEqn(const LinearEqn &eqn)
		: m(eqn.m), b(eqn.b), z(eqn.z) { } // copy constructor
	~LinearEqn() { } // destructor

	float y(float x) {
		return m*x + b;
	}

	// Ignores potential z-height errors..
	Point3f Intersection(LinearEqn eqn) {
		float x = (eqn.b - this->b) / (this->m - eqn.m);
		float y = this->m*x + this->b;
		return Point3f(x,y,z);
	}

	void Rotate(const float phiRad) {
		float mp = this->m + phiRad;

		float xp = -this->b*sin(phiRad);
		float yp =  this->b*cos(phiRad);

		float bp = yp-mp*xp;

		this->m = mp;
		this->b = bp;
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
	T get(const int y, const int x) const {
		return vmap.at(y).at(x);
	}
//	void doFunc(VectorMap<Point2f>& toMap, void (*func)(VectorMap&, VectorMap&, int, int) ) { //B //
//		for (int h=0; h < height; ++h) {
//			for (int w=0; w < width; ++w) {
//				func(*this,toMap,h,w); //
//			}
//		}
//	}
};

//void funcDoIntersections(VectorMap<LinearEqn>& from, VectorMap<Point2f>& to, int h, int w) { // to
//	LinearEqn curr = from.get(h,w);
//	LinearEqn next = from.get(h,(w+1)%from.width);
//
//	Point2f intersection = curr.Intersection(next);
//	to.set(h,w,intersection);
//}

void DoIntersection(VectorMap<LinearEqn>& from, VectorMap<Point3f>& to) {
	LinearEqn curr (0,0,0);
	LinearEqn next (0,0,0);
	for (int h=0; h < from.height; ++h) {
		for (int w=0; w < from.width; ++w) {
			curr = from.get(h,w);
			next = from.get(h,(w+1)%from.width);

			Point3f intersection = curr.Intersection(next);
			to.set(h,w,intersection);
		}
	}
}

void DoMidpoints(VectorMap<Point3f>& from, VectorMap<Point3f>& to) {
	Point3f curr (0,0,0);
	Point3f next (0,0,0);
	for (int h=0; h < from.height; ++h) {
		for (int w=0; w < from.width; ++w) {
			curr = from.get(h,w);
			next = from.get(h,(w+1)%from.width);

			Point3f midpoint = Point3f( (curr.x + next.x)/2,
					(curr.y + next.y)/2,
					(curr.z + next.z)/2);
			to.set(h,w,midpoint);
		}
	}
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
		//for (int c=0; c < hsv.cols; c++) {
		// don't look any farther than center column
		for (int c=0; c < CENTERCOL; c++) {
			//hue from 205-265, saturation > 50%, value > 30% ?
			Vec3b pt = hsv.at<Vec3b>(r,c);

			//why the hell is the hue value so small? It never goes above 17.
			if (int(pt.val[0]) > 100 && int(pt.val[0]) < 135 &&
					int(pt.val[1]) > 125 &&
					int(pt.val[2]) > 100) {
				//save point
				ImagePoint vert (CENTERCOL - c,r);
				edges.push_back(vert);
				//edges.at<Point>(i) = vert;

				//skip to next row
				found = true;
				break;
			}
		}
		if (!found) { edges.push_back(ImagePoint(0,r)); }

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

//generates two triangle verts and indices for each square.
//essentially translates the 2d Vertex matrix into a 1D matrix + index values.
void OrientRadialTrisVerts(VectorMap<Point3f>& pts, Vector<Point3f>& verts, Vector<Point3i>& tris) {
//	Point3f pt1 (0,0,0);
//	Point3f pt2 (0,0,0);
//	Point3f pt3 (0,0,0);
//	Point3f pt4 (0,0,0);

	//add each point to verts for each itteration.
	for (int h=0; h < pts.height; ++h) {
		for (int w=0; w < pts.width; ++w) {
			verts.push_back(pts.get(h,w));
		}
	}

	//height levels first...
	for (int h=0; h < pts.height-1; ++h) {
		for (int w=0; w < pts.width; ++w) {
//			pt1 = pts.get(h,w); 					//top left
//			pt2 = pts.get(h,(w+1)%pts.width);		//top right, width wraps around object
//			pt3 = pts.get(h+1,w);					//bottom left
//			pt4 = pts.get(h+1,(w+1)%pts.width);		//bottom right, width wraps around object

			int i1 = (h*pts.height)+w;						//top left
			int i2 = (h*pts.height)+((w+1)%pts.width);		//top right, width wraps around object
			int i3 = ((h+1)*pts.height)+w;					//bottom left
			int i4 = ((h+1)*pts.height)+((w+1)%pts.width);	//bottom right, width wraps around object

			//add 3 verts for each triangle
			tris.push_back(Point3i(i1,i3,i2));
			tris.push_back(Point3i(i2,i3,i4));
		}
	}
}

#include <iostream>
#include <fstream>
#include <ctime>

using std::ofstream;

void WriteVerts(Vector<Point3f>& verts, ofstream& out, float s) {
	for (auto vert: verts) {
		out << "v " << vert.x*s << " " << vert.y*s << " " << vert.z*s << "\n";
	}
}

void WriteTris(Vector<Point3i>& tris, ofstream& out) {
	for (auto tri: tris) {
		out << "f " << tri.x << " " << tri.y << " " << tri.z << "\n";
	}
}

int write (string filename, string desc, Vector<Point3f>& verts, Vector<Point3i>& tris, float s) {
  ofstream myfile;
  myfile.open (filename);

  //write header:
  time_t t = time(0);
  struct tm * now = localtime( & t );
      cout << (now->tm_year + 1900) << '-'
           << (now->tm_mon + 1) << '-'
           <<  now->tm_mday
           << endl;

  myfile << "Filename: " << filename << "\n";
  myfile << "Date: " << (now->tm_year + 1900) << '-'
          << (now->tm_mon + 1) << '-'
          <<  now->tm_mday << "\n";
  myfile << "Description: " << desc << "\n\n";

  //write tris and verts
  WriteVerts(verts,myfile,s);
  myfile << "\n";
  WriteTris(tris,myfile);

  myfile.close();
  return 0;
}


int main( int argc, char** argv ) {

	vector<string> filenames = {"0.jpg", "30.jpg", "60.jpg",
			"90.jpg", "120.jpg", "150.jpg",
			"240.jpg", "270.jpg", "300.jpg", "330.jpg"};

	Mat image;
	//image = imread( argv[1], 1 );
	Rect roi (1000,1000,500,500); //user input
	int CENTERCOL = 417; //user input or determined by machine architecture
	int NUMANGLES = filenames.size();
	int NUMVERTSTEPS = 10;
	float FOCALDIST = 2000;

	AnglePointMap apMap; //used because I don't know how big this is??
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

		vector<AnglePoint> anglePoints;
		anglePoints = getThetas(edges, FOCALDIST, phi);
		phi = phi + 30.0;

		apMap.push_back(anglePoints);

		//dispEdge(edges, image_roi, CENTERCOL);
	}
	//VectorMap<ImagePoint> edgesMap (NUMVERTSTEPS, NUMANGLES, ImagePoint(0,0));
	//cout << "NumVertSteps: " << NUMVERTSTEPS << " " << apMap.size() << endl;
	//cout << "NUMANGLES: " << NUMANGLES << " " << apMap.at(0).size() << endl;


	//now have a 2d vector of AnglePoints. Build the map.
	//VectorMap considers the texture as rolled off. Need to transpose the AnglePoints.
	//This is starting to look an awfull lot like i'm just writing a linear algebra library.
	VectorMap<LinearEqn> eqns (NUMVERTSTEPS, NUMANGLES, LinearEqn(0,0,0));
	for (int h=0; h<NUMVERTSTEPS; ++h) {
		for (int w=0; w<NUMANGLES; ++w) {
			AnglePoint pt = apMap.at(w).at(h);
			float m = tan(pt.theta);
			float b = m*FOCALDIST;
			LinearEqn thiseqn (m,b,pt.z);

			//rotate the equation about the origin...
			//actually rotate the points, then build the equation.
			thiseqn.Rotate(pt.phi * PI / 180);

			eqns.set(h, w, thiseqn);
		}
	}
	// where is the angle phi incorporated??

	//Now have a map of linear equations.. no obvious float range problems so far.
	//Build Intersection Points? Can I do this in the existing map?
	//Give LinearEqn knowledge of its adjacent item?
	VectorMap<Point3f> intersections (NUMVERTSTEPS, NUMANGLES, Point3f(0,0,0));;
	//Beqns.doFunc(intersections, &funcDoIntersections); //;, intersections
	DoIntersection(eqns, intersections);

	VectorMap<Point3f> midpoints (NUMVERTSTEPS, NUMANGLES, Point3f(0,0,0));;
	DoMidpoints(intersections, midpoints);


	//compute
	Vector<Point3f> verts;
	Vector<Point3i> tris;
	OrientRadialTrisVerts(midpoints, verts, tris);

	//write to file
	write ("out.obj", "test cylinders", verts, tris, 1./300.);

//	dispEdge(edges, image_roi, centerCol);
//	Mat clipped;
//	inRange(hsv, Scalar(100, 100, 100), Scalar(135,255,255), clipped);
//
//	dispImage(clipped);

	return 0;
}

