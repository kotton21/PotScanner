/*
 * test2.cpp
 *
 *  Created on: Jan 10, 2017
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
	float phi;
	ImagePoint(int x, int y, float phi) {
		this->x = x;
		this->y = y;
		this->phi = phi;
	}
	ImagePoint(const ImagePoint& obj) : ImagePoint(obj.x, obj.y, obj.phi) { }
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
		float thetap = atan(this->m);
		float mp = tan(thetap + phiRad);

		float xp = -this->b*sin(phiRad);
		float yp =  this->b*cos(phiRad);

		float bp = yp-mp*xp;

		this->m = mp;
		this->b = bp;
	}
};

//a 2d map of any type. must provide own functionality for resizing
//fills by rows down
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
	void set(const int y, vector<T>& vec) {
		vmap.at(y) = vec;
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



vector<ImagePoint> detectEdges(const Mat hsv, const int NUMVERTSTEPS, const int CENTERCOL, float phi) {
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
				ImagePoint vert (CENTERCOL - c, r, phi);
				edges.push_back(vert);
				//edges.at<Point>(i) = vert;

				//skip to next row
				found = true;
				break;
			}
		}
		if (!found) { edges.push_back(ImagePoint(0, r, phi)); }

		i = i + 1;
		r = r + pixelstep;
	}
	return edges;
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
	for (int v=0; v < pts.width; ++v) {
		for (int a=0; a < pts.height; ++a) {

			verts.push_back(pts.get(a,v));
		}
	}

	//height levels first...
	for (int v=0; v < pts.width-1; ++v) {
		for (int a=0; a < pts.height; ++a) {
//			int i1 = 1+(a*pts.height)+v;						//top left
//			int i2 = 1+(a*pts.height)+((v+1)%pts.width);		//top right, width wraps around object
//			int i3 = 1+((a+1)*pts.height)+v;					//bottom left
//			int i4 = 1+((a+1)*pts.height)+((v+1)%pts.width);	//bottom right, width wraps around object

			int i1 = 1+(v*pts.height)+a;						//top left
			int i2 = 1+(v*pts.height)+((a+1)%pts.width);		//top right, width wraps around object
			int i3 = 1+((v+1)*pts.height)+a;					//bottom left
			int i4 = 1+((v+1)*pts.height)+((a+1)%pts.width);	//bottom right, width wraps around object

			//add 3 verts for each triangle
			tris.push_back(Point3i(i1,i3,i2));
			tris.push_back(Point3i(i2,i3,i4));
		}
	}
}

void ScaleVerts(Vector<Point3f>& verts, const float s) {
	for (unsigned int i = 0; i < verts.size(); ++i) {
		verts[i] = verts[i]*s;
	}
}

#include <iostream>
#include <fstream>
#include <ctime>

using std::ofstream;

void WriteVerts(Vector<Point3f>& verts, ofstream& out) {
	for (auto vert: verts) {
		out << "v " << vert.x << " " << vert.y << " " << vert.z << "\n";
	}
}

void WriteTris(Vector<Point3i>& tris, ofstream& out) {
	for (auto tri: tris) {
		out << "f " << tri.x << " " << tri.y << " " << tri.z << "\n";
	}
}

int write (string filename, string desc, Vector<Point3f>& verts, Vector<Point3i>& tris) {
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
  WriteVerts(verts,myfile);
  myfile << "\n";
  WriteTris(tris,myfile);

  myfile.close();
  return 0;
}



int main( int argc, char** argv ) {

	vector<string> filenames = {"0.jpg", "30.jpg", "60.jpg",
			"90.jpg", "120.jpg", "150.jpg", "180.jpg", "210.jpg",
			"240.jpg", "270.jpg", "300.jpg", "330.jpg"}; //


	Mat image;
	//image = imread( argv[1], 1 );
	Rect roi (1000,1000,500,500); //user input
	int CENTERCOL = 417; //user input or determined by machine architecture
	int NUMANGLES = filenames.size();
	int NUMVERTSTEPS = 13;
	float FOCALDIST = 1000;

	cout << "NUMANGLES: " << NUMANGLES << endl; 		//vector height/angle
	cout << "NUMVERTSTEPS: " << NUMVERTSTEPS << endl;	//vector width/image step


	//First index is the angle, 2nd index is the height... wtf.
	VectorMap<ImagePoint> ipMap (NUMANGLES, NUMVERTSTEPS, ImagePoint(0,0,0));
	float phi = 0;
	for (unsigned int i = 0; i < filenames.size(); ++i) {
		// Region of Interest
		image = imread( "pics/"+filenames[i], 1 );
		Mat image_roi(image, Rect(2500, 500, 800, 600)); // x0, y0, w, h
		Mat hsv;
		cvtColor(image_roi, hsv, CV_BGR2HSV);

		vector<ImagePoint> edges = detectEdges(hsv, NUMVERTSTEPS, CENTERCOL, phi);

		phi = phi + 30.0;

		ipMap.set(i,edges); //i is dim 1, which is the angle...

		//dispEdge(edges, image_roi, CENTERCOL);

		//for (unsigned int i=0; i<edges.size(); ++i) {
		//	cout << edges[i].x << "," << edges[i].y << "," << edges[i].phi << endl;
		//}

	}

	//now have a 2d vector of AnglePoints. Build the map.
	//VectorMap considers the texture as rolled off. Need to transpose the AnglePoints.
	//This is starting to look an awfull lot like i'm just writing a linear algebra library.
	VectorMap<LinearEqn> eqns (NUMANGLES, NUMVERTSTEPS, LinearEqn(0,0,0)); //this constructor is useless here.
	for (int a=0; a<NUMANGLES; ++a) {
		for (int v=0; v<NUMVERTSTEPS; ++v) {
			ImagePoint pt = ipMap.get(a,v);
			//float m = pt.x/FOCALDIST;
			//float b = pt.x; // y-mx
			//LinearEqn thiseqn (m, b, pt.y);

			float theta = atan2(pt.x, FOCALDIST);
			float theta_phi = theta + (pt.phi * PI / 180);
			float b = pt.x; // x = 0, y = b
			float x_phi = b*(-sin(pt.phi * PI / 180));
			float y_phi = b*(cos(pt.phi * PI / 180));
			float m_phi = tan(theta_phi);
			float b_phi = y_phi - (m_phi * x_phi); // y-mx = b

			cout << "phi: " << pt.phi << " deg: " << pt.phi *PI/180 << " theta " << theta << " theta_phi " << theta_phi << endl;

			LinearEqn thiseqn (m_phi,b_phi,pt.y);

			//rotate the equation about the origin...
			//thiseqn.Rotate(pt.phi * PI / 180);

			eqns.set(a, v, thiseqn);
		}
	}

	//Now have a map of linear equations.. no obvious float range problems so far.
	//Give LinearEqn knowledge of its adjacent item?
	VectorMap<Point3f> intersections (NUMANGLES, NUMVERTSTEPS, Point3f(0,0,0));;
	//DoIntersection(eqns, intersections);
	LinearEqn curr (0,0,0);
	LinearEqn next (0,0,0);
	for (int a=0; a<NUMANGLES; ++a) {
		for (int v=0; v<NUMVERTSTEPS; ++v) {
			curr = eqns.get(a,v);
			next = eqns.get((a+1)%NUMANGLES,v);

			Point3f intersection = curr.Intersection(next);
			intersections.set(a,v,intersection);
		}
	}


	VectorMap<Point3f> midpoints (NUMANGLES, NUMVERTSTEPS, Point3f(0,0,0));
//	DoMidpoints(intersections, midpoints);
	Point3f currp (0,0,0);
	Point3f nextp (0,0,0);
	for (int a=0; a<NUMANGLES; ++a) {
		for (int v=0; v<NUMVERTSTEPS; ++v) {
			currp = intersections.get(a,v);
			nextp = intersections.get((a+1)%NUMANGLES,v);

			Point3f midpoint = Point3f( (currp.x + nextp.x)/2,
					(currp.y + nextp.y)/2, currp.z );
					//(currp.z + nextp.z)/2);
			midpoints.set(a,v,midpoint);
		}
	}


	//compute
	Vector<Point3f> verts;
	Vector<Point3i> tris;
	OrientRadialTrisVerts(midpoints, verts, tris);

	ScaleVerts(verts, 1./400.);

	//write to file
	string path = "/home/karl/Documents/COMSW4160/COMS_4160/hw2.2/";
	write (path + "out.obj", "test cylinders stuff", verts, tris);

//	dispEdge(edges, image_roi, centerCol);
//	Mat clipped;
//	inRange(hsv, Scalar(100, 100, 100), Scalar(135,255,255), clipped);
//
//	dispImage(clipped);

	return 0;
}
