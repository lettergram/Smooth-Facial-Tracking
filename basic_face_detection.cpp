#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;
String window_name = "Face Detection";

/**
 * Detects faces and draws an ellipse around them
 */
void detectFaces(Mat frame) {

  std::vector<Rect> faces;
  Mat frame_gray;

  // Convert to gray scale
  cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

  // Equalize histogram
  equalizeHist(frame_gray, frame_gray);

  // Detect faces
  face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3,
				0|CASCADE_SCALE_IMAGE, Size(30, 30));

  // Iterate over all of the faces
  for( size_t i = 0; i < faces.size(); i++ ) {

    // Find center of faces
    Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);

    // Draw ellipse around face
    ellipse(frame, center, Size(faces[i].width/2, faces[i].height/2),
	    0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

  }

  // Display frame
  imshow( window_name, frame );
}


int main() {
  
  VideoCapture cap(0); // Open default camera
  Mat frame;

  // Load preconstructed classifier
  face_cascade.load("haarcascade_frontalface_alt.xml");
  
  while(cap.read(frame)) {
    detectFaces(frame); // Call function to detect faces
    if( waitKey(30) >= 0)    // pause
      break; 
  }
  return 0;
}
