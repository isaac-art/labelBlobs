#include "ofApp.h"
using namespace cv;
//--------------------------------------------------------------
void ofApp::setup(){	
    ofBackground(230, 230, 230);
    
    //CV
    video.setDeviceID(0);//0 back 1 front
    video.setDesiredFrameRate(30);
    video.initGrabber(360,480);
    
    //ML
    guiGeneral.setup("general", "general", 10, 350);
    guiGeneral.setDefaultHeight(40);
    guiGeneral.add(modelControl.setup("run model", false));
    guiGeneral.add(trainThis.setup("train"));
    guiGeneral.add(resetModel.setup("reset model"));
    
    trained = false;
    resetModel.addListener(this, &ofApp::resetModelPressed);
    trainThis.addListener(this, &ofApp::trainTheModel);
}

//--------------------------------------------------------------
void ofApp::update(){
    video.update();
    if(video.isFrameNew()){
        image.setFromPixels(video.getPixelsRef());
        grayImage = image;
        blurred = grayImage;
        blurred.blur(5);
        mask = blurred;
        mask.invert();
        mask.threshold(145);//ofMap(mouseY,0,ofGetHeight(),0,200)
        mask.erode();
        mask.dilate();
        contourFinder.findContours( mask, 20, 20000, 5, true, true);
        blobs = contourFinder.blobs;
    }
    
    
    if (recordingState > 0 && blobs.size() == 5) {
        trainingExample tempExample;
        for(int i = 0; i < blobs.size();i++){
            ofVec2f cent = blobs[i].centroid;
            tempExample.input.push_back(cent.x);
            tempExample.input.push_back(cent.y);
            tempExample.output.push_back(cent.x);
            tempExample.output.push_back(cent.y);
        }
        trainingSet.push_back(tempExample);
    }
    else if (trained && modelControl == 1 && blobs.size() == 5) {
        output.clear();
        vector<double> inputVec;
        for(int i = 0; i < blobs.size();i++){
            ofVec2f cent = blobs[i].centroid;
            inputVec.push_back(cent.x);
            inputVec.push_back(cent.y);
        }
        output = myNN.run(inputVec);
    }
}
void ofApp::resetModelPressed() {
    std::cout << "resetting models" << std::endl;
    myNN.reset();
    trainingSet.clear();
    modelControl = false;
}
//--------------------------------------------------------------
void ofApp::draw(){
    ofSetColor(255, 255, 255, 80); //for video
    image.draw(0,0);
    ofPushStyle();
    for(int i=0; i<blobs.size(); i++){
        ofSetColor(220,10,10,100);
        ofSetLineWidth(3);
        blobs[i].draw();
        ofSetColor(245,245,245);
        //int sz = blobs[i].nPts;
        //ofPoint center = blobs[i].centroid;
        //ofDrawBitmapString("thing", center.x, center.y);
    }
    
    
    for(int j = 0; j < output.size();j+=2){
        ofDrawBitmapString("thing", output[j], output[j+1]);
    }
    ofPopStyle();
    
    
    guiGeneral.draw();
    
}

//--------------------------------------------------------------
void ofApp::exit(){

}

void ofApp::trainTheModel(){
    if(recordingState == 1){
        recordingState = 0;
        if (trainingSet.size() > 0) {
            trained = myNN.train(trainingSet);
            //std::cout << "trained: " << trained << std::endl;
        }
    }else{
        recordingState = 1;
    }
}

//--------------------------------------------------------------
void ofApp::touchDown(ofTouchEventArgs & touch){

}

//--------------------------------------------------------------
void ofApp::touchMoved(ofTouchEventArgs & touch){

}

//--------------------------------------------------------------
void ofApp::touchUp(ofTouchEventArgs & touch){

    
}

//--------------------------------------------------------------
void ofApp::touchDoubleTap(ofTouchEventArgs & touch){

}

//--------------------------------------------------------------
void ofApp::touchCancelled(ofTouchEventArgs & touch){
    
}

//--------------------------------------------------------------
void ofApp::lostFocus(){

}

//--------------------------------------------------------------
void ofApp::gotFocus(){

}

//--------------------------------------------------------------
void ofApp::gotMemoryWarning(){

}

//--------------------------------------------------------------
void ofApp::deviceOrientationChanged(int newOrientation){

}
