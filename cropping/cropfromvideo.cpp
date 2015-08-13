/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"


#include <iostream>
#include <fstream>
#include <sstream>

// -lopencv_core -lopencv_objdetect -lopencv_highgui -lopencv_imgproc

using namespace cv;
using namespace std;


    string face_cascade_name = "lbpcascade_frontalface.xml";
    CascadeClassifier face_cascade;
    string window_name = "Capture - Face detection";
    int filenumber; // Number of file to be saved
    string filename, filename1;

    void detectAndDisplay(Mat frame);


int main() {
    
     VideoCapture capture("surprise.mkv");

    if (!capture.isOpened())  // check if we succeeded
      return -1;

    // Load the cascade
    if (!face_cascade.load(face_cascade_name))
    {
        printf("--(!)Error loading\n");
        return (-1);
    };
   
    Mat frame;

    for(;;)
    {

        capture >> frame;

   	    if (!frame.empty())
        {
            detectAndDisplay(frame);
        }
         
        waitKey(10);
        
    }
    return 0;
    
}


// Function detectAndDisplay
void detectAndDisplay(Mat frame)
{
    
    std::vector<Rect> faces;
    Mat frame_gray;
    Mat crop;
    Mat res;
    Mat gray;
    string text;
    stringstream sstm,sstd;
    Mat raisa;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    // Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    // Set Region of Interest
    cv::Rect roi_b;
    cv::Rect roi_c;

    size_t ic = 0; // ic is index of current element
    int ac = 0; // ac is area of current element

    size_t ib = 0; // ib is index of biggest element
    int ab = 0; // ab is area of biggest element

    for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)

    {
        roi_c.x = faces[ic].x;
        roi_c.y = faces[ic].y;
        roi_c.width = (faces[ic].width);
        roi_c.height = (faces[ic].height);

        ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)

        roi_b.x = faces[ib].x;
        roi_b.y = faces[ib].y;
        roi_b.width = (faces[ib].width);
        roi_b.height = (faces[ib].height);

        crop = frame(roi_b);
        resize(crop, res, Size(128, 128), 1, 1, INTER_LINEAR); // This will be needed later while saving images
        cvtColor(res, gray, CV_BGR2GRAY);

        
        // masking program 
       

        // Form a filename
        filename = "//dataset/";
        stringstream ssfn;
        ssfn << filenumber << ".jpg";
        filename = ssfn.str();
        filenumber++;
        cv::imwrite(filename,gray);

        filename1 = "//dataset";
        stringstream sstd;
        sstd << filenumber << ".jpg";
        filename1 = sstd.str();
        filenumber++;
        cv::imwrite(filename1,raisa);

       


       
        Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
        Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
        rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
    }


        Mat im2(gray.rows, gray.cols, CV_8UC1, Scalar(0,0,0));
        ellipse( im2, Point( 64,64 ), Size( 50.0, 60.0 ), 0, 0, 360, Scalar( 255, 255, 255), -1, 8 );
        bitwise_and(gray,im2,raisa); 


    // Show image
    sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << filename;
    text = sstm.str();


    if (!crop.empty())
    {
        imshow("detected", crop);
        imwrite(filename,raisa);
    }
    else
        destroyWindow("detected");
        imwrite(filename1,raisa);
        imwrite(filename, frame);

}
