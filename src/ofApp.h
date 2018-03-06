#pragma once

#include "ofxiOS.h"
#include "ofxRapidLib.h"
#include "ofxOpenCv.h"
#include "ofxGui.h"
using namespace cv;

class ofApp : public ofxiOSApp {
	
    public:
        void setup();
        void update();
        void draw();
        void exit();
	
        void touchDown(ofTouchEventArgs & touch);
        void touchMoved(ofTouchEventArgs & touch);
        void touchUp(ofTouchEventArgs & touch);
        void touchDoubleTap(ofTouchEventArgs & touch);
        void touchCancelled(ofTouchEventArgs & touch);

        void lostFocus();
        void gotFocus();
        void gotMemoryWarning();
        void deviceOrientationChanged(int newOrientation);
    
        //ML
        void resetModelPressed();
        void trainTheModel();
        ofxToggle modelControl;
        ofxButton resetModel;
        ofxButton trainThis;
        ofxPanel guiGeneral;
    
        regression myNN;
        vector<trainingExample> trainingSet;
        int recordingState;
        bool trained;
        vector<double> output;
    
        //CV
        ofVideoGrabber video;
        ofxCvColorImage image;
        ofxCvContourFinder contourFinder;
        ofxCvGrayscaleImage grayImage;
        ofxCvGrayscaleImage blurred;
        ofxCvGrayscaleImage background;
        ofxCvGrayscaleImage diff;
        ofxCvGrayscaleImage mask;
        vector<ofxCvBlob> blobs;
    
};


