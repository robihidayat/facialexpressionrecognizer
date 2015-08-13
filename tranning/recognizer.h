#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/ml/ml.hpp> // library svm

using namespace std;
using namespace cv;


static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

static void dbread(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}



void eigenFaceTrainer(){

    ofstream myfile1;
    myfile1.open("datatrainingEigenface.txt");
    string output_folder = "eigenface"; // inisialisasi folder untuk menyimpan gambar transformasi

    vector<Mat> images; // inisialisasi images sebagai vektor, maktiks
    vector<int> labels; // inisialisasi label sebagai vektor , integer

    try {

        string filename = "dataset_jaffe.csv"; // inisialisai filename sebagai dataset.csv
        dbread(filename, images, labels); //  menggunakan fungsi dbread untuk membaca csv dile.
        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout <<"////////////////////////TRAINING METODE FisherFace//////////////////////////////"<<endl;
        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout<<"size of the images is "<<images.size()<<endl; // menampilkan ukuran dari gambar
        cout<<"size of the label is "<<labels.size()<<endl; // menapilkan ukuran label

        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout<<"...............................training begins........................"<<endl; // menampilkan teks.
    }
    catch(cv::Exception& e){
        cerr<<"error opening file dab...."<<e.msg<<endl;
        exit(1);
    }

    Ptr<FaceRecognizer> model = createEigenFaceRecognizer(); // inisialisasi model sebagai ??? 
  

    double t = (double)getTickCount();
    model ->train(images,labels);
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Waktu Training Metode Eigenfaces: " << t << endl;
    myfile1<< "Waktu Training Metode Eigenfaces: " << t << endl;

    int height = images[0].rows;
    int w = images[0].cols;

    cout << "w"<<w<<"H"<<height<<endl;
    model->save("saveEigen.yml");

    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout<<"........................Training Finish ................................"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;

    Mat eigenvalues = model -> getMat("eigenvalues");
    //and we can do the same to display the eigenvector (read eigenfaces)
    Mat W = model -> getMat("eigenvectors");
    //Get the sample mean from the traing data
    Mat mean = model -> getMat("mean");
   
    //imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
    imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));

   // Display or save the Eigenfaces:
    for (int i = 0; i < min(150, W.cols); i++) {
        
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        myfile1<<msg<<endl;
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // Reshape to original size & normalize to [0...255] for imshow.
        Mat grayscale = norm_0_255(ev.reshape(1, height));
        // Show the image & apply a Jet colormap for better sensing.
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
        // Display or save:
       // imshow(format("eigenface_%d", i), cgrayscale);
        imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
        
    }
    myfile1.close();
    // Display or save the image reconstruction at some predefined steps:
    for(int num_components = min(W.cols, 10); num_components < min(W.cols, 300); num_components+=15) {
        // slice the eigenvectors from the model
        Mat evs = Mat(W, Range::all(), Range(0, num_components));
        Mat projection = subspaceProject(evs, mean, images[0].reshape(1,1));
        Mat reconstruction = subspaceReconstruct(evs, mean, projection);
        // Normalize the result:
        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
        imwrite(format("%s/eigenface_reconstruction_%d.png", output_folder.c_str(), num_components), reconstruction);
        // Display or save:
        
    }

}



void FisherFaceTrainer(){
    ofstream myfile2;
    myfile2.open("datatrainingFisherface.txt");
    string output_folder = "fisherfacerec"; // inisialisasi folder untuk menyimpan gambar transformasi

    vector<Mat> images; // inisialisasi images sebagai vektor, maktiks
    vector<int> labels; // inisialisasi label sebagai vektor , integer

    try {
        string filename = "dataset_jaffe.csv"; // inisialisai filename sebagai dataset.csv
        dbread(filename, images, labels); //  menggunakan fungsi dbread untuk membaca csv dile.
        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout <<"////////////////////////TRAINING METODE EIGENFACE//////////////////////////////"<<endl;
        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout<<"size of the images is "<<images.size()<<endl; // menampilkan ukuran dari gambar
        cout<<"size of the label is "<<labels.size()<<endl; // menapilkan ukuran label
        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout<<"...............Training Metode FisherFace begins......................."<<endl;
    }
    catch(cv::Exception& e){
        cerr<<"error opening file ...."<<e.msg<<endl;
        exit(1);
    }

    int im_width = images[0].cols;
    int im_height = images[0].rows;

    
    Ptr<FaceRecognizer> model2 = createFisherFaceRecognizer(); // inisialisasi model sebagai ??? 
    
    double v = (double)getTickCount();
    model2 ->train(images,labels);
    v = ((double)getTickCount() - v)/getTickFrequency();
    cout << "Waktu Training Metode FisherFace: " << v << endl;
    myfile2 << "Waktu Training Metode FisherFace: " << v << endl;
    int height = images[0].rows;
    model2->save("saveFisher.yml");
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout<<"........................Training Finish ................................"<<endl;

    Mat eigenvalues = model2 -> getMat("eigenvalues");
    //and we can do the same to display the eigenvector (read eigenfaces)
    Mat W = model2 -> getMat("eigenvectors");
    //Get the sample mean from the traing data
    Mat mean = model2 -> getMat("mean");


   // imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
    imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
    
   // Display or save the Eigenfaces:
    for (int i = 0; i < min(150, W.cols); i++) 
    {
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        myfile2<<msg<<endl;
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // Reshape to original size & normalize to [0...255] for imshow.
        Mat grayscale = norm_0_255(ev.reshape(1, height));
        // Show the image & apply a Jet colormap for better sensing.
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
        // Display or save:
       // imshow(format("eigenface_%d", i), cgrayscale);
        imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
        
    }
    myfile2.close();
    // Display or save the image reconstruction at some predefined steps:
    for(int num_component = 0; num_component < min(16, W.cols); num_component++) {
        // Slice the Fisherface from the model:
        Mat ev = W.col(num_component);
        Mat projection = subspaceProject(ev, mean, images[0].reshape(1,1));
        Mat reconstruction = subspaceReconstruct(ev, mean, projection);
        // Normalize the result:
        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
        // Display or save:
       imwrite(format("%s/fisherface_reconstruction_%d.png", output_folder.c_str(), num_component), reconstruction);
        }
    

}

void LBPHFaceTrainer()
{
    ofstream myfile3;
    myfile3.open("datatrainingLBPH.txt");

    vector<Mat> images;
    vector<int> labels;

    try {
        string filename = "dataset_jaffe.csv";
        dbread(filename, images, labels);
        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout <<"////////////////////////TRAINING METODE EIGENFACE//////////////////////"<<endl;
        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout<<"size of the images is "<<images.size()<<endl;
        cout<<"size of the label is "<<labels.size()<<endl;
        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
        cout<<"..........................Training Metode LBPH ........................"<<endl;
    }

    catch(cv::Exception& e){
        cerr<<"error opening file ...."<<e.msg<<endl;
        exit(1);
    }

    Ptr<FaceRecognizer> model1 = createLBPHFaceRecognizer();

    double y = (double)getTickCount();
    model1 ->train(images,labels);
    y = ((double)getTickCount() - y)/getTickFrequency();
    cout << "Waktu Training Metode LBPH: " << y << endl;
    myfile3<<"Waktu Training Metode LBPH: " << y << endl;
    int height = images[0].rows;
    model1->save("saveLBPH.yml");

    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout<<"........................Training Finish ................................"<<endl;
    cout << "Model Information:" << endl;
    string model1_info = format("\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
            model1->getInt("radius"),
            model1->getInt("neighbors"),
            model1->getInt("grid_x"),
            model1->getInt("grid_y"),
            model1->getDouble("threshold"));
    cout << model1_info << endl;
    myfile3 <<model1_info << endl;

    // We could get the histograms for example:
    vector<Mat> histograms = model1->getMatVector("histograms");
    // But should I really visualize it? Probably the length is interesting:
    cout << "Size of the histograms: " << histograms[0].total() << endl;
    myfile3<<"Size of the histograms: " << histograms[0].total() << endl;
    myfile3.close();
    

}




string g_listname_t[] =
{
    
    "sedih",
    "senang",
    "kaget",
    "netral"
    
};


int EigenRecognition()
{
    ofstream myfile4;
    myfile4.open("dataUJIEigenFace.txt");
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"...................start Recognizeing Eigenface...."<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
   
    vector<Mat> images;
    vector<int> labels;

    try {
        string filename = "datauji_jaffe.csv";
        dbread(filename, images, labels);
        myfile4<<"size of the images is "<<images.size()<<endl;
        myfile4<<"size of the label is "<<labels.size()<<endl;
        cout<<"size of the images is "<<images.size()<<endl;
        cout<<"size of the label is "<<labels.size()<<endl;
        }
    catch(cv::Exception& e){
        cerr<<"error opening file dab...."<<e.msg<<endl;
        exit(1);
    }
    
    int im_width = images[0].cols;
    int im_height = images[0].rows;

   

    int iii;
    for(iii=1;iii<images.size();iii++){
    cout<<"sample ke "<<iii<<endl;
    Mat testSample = images[images.size() - iii];
    int testLabel = labels[labels.size() - iii];

   // images.pop_back();
    //labels.pop_back();

    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    model -> load ("saveEigen.yml");

    int predictedLabel = -1;
    double confidence = 0.0;

    double q = (double)getTickCount();
    model->predict(testSample, predictedLabel, confidence) ;
    q = ((double)getTickCount() - q)/getTickFrequency();
    cout << "Times passed in seconds: " << q << endl;
    cout<<"confidence "<<confidence<<endl;
    string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    cout << result_message <<" "<<"confidence"<< confidence<<endl;
    myfile4<<"Data Uji Ke= "<<images.size() - iii <<";Prediksi= "<<predictedLabel<<";Actualclass= "<<testLabel<<" ;Kecepatan= "<<q<<" ;confidence= "<<confidence<<endl;
   // myfile4<<result_message << endl;
     cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
   imshow("Eigenface",testSample);
   waitKey(10);
   }
      myfile4.close();
}



int FisherRecognition()
{
    ofstream myfile5;
    myfile5.open("dataUJIFisherface.txt");
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"start Recognizeing Fisherface...."<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
   
    vector<Mat> images;
    vector<int> labels;

    try {
        string filename = "datauji_jaffe.csv";
        dbread(filename, images, labels);

        cout<<"size of the images is "<<images.size()<<endl;
        cout<<"size of the label is "<<labels.size()<<endl;
        }
    catch(cv::Exception& e){
        cerr<<"error opening file dab...."<<e.msg<<endl;
        exit(1);
    }
    
  //  int im_width = images[0].cols;
   // int im_height = images[0].rows;

    int l;
    for (l=1;l<images.size();l++) 
    {
        cout <<"pengujian ke : "<<l<<endl;

    
    Mat testSample = images[images.size() -l];
    int testLabel = labels[labels.size() -l];

   // images.pop_back();
    //labels.pop_back();


    Ptr<FaceRecognizer> model2 = createFisherFaceRecognizer();
    model2 -> load ("saveFisher.yml");


    int predictedLabel = -1;
    double confidence = 0.0;
   

    double z = (double)getTickCount();
    model2->predict(testSample, predictedLabel, confidence);
    z = ((double)getTickCount() - z)/getTickFrequency();
    cout << "Times passed in seconds: " << z << endl;
   
    string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    cout << result_message << endl;
    myfile5<<"Data Uji Ke= "<<images.size() - l <<";Prediksi= "<<predictedLabel<<";Actualclass= "<<testLabel<<";Kecepatan= "<<z<<";confidence= "<<confidence<<endl;
    cout<<"confidence "<<confidence<<endl;
     cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
   imshow("FisherFace",testSample);
    waitKey(10); 

    }

      myfile5.close();
}


int LBPHRecognition()
{
    ofstream myfile6;
    myfile6.open("dataUJILBPH.txt");
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"start Recognizeing LBPH...."<<endl;
     cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    vector<Mat> images;
    vector<int> labels;

    try {
        string filename = "datauji_jaffe.csv";
        dbread(filename, images, labels);

        cout<<"size of the images is "<<images.size()<<endl;
        cout<<"size of the label is "<<labels.size()<<endl;
        }
    catch(cv::Exception& e){
        cerr<<"error opening file dab...."<<e.msg<<endl;
        exit(1);
    }
    
    int im_width = images[0].cols;
    int im_height = images[0].rows;

    

int a;
for (a=1;a<images.size();a++)
    {
    Ptr<FaceRecognizer> model1 = createLBPHFaceRecognizer();
    model1 -> load ("saveLBPH.yml");
   

    

    cout <<"Waktu Training DataSet : " << a << endl;
    Mat testSample = images[images.size() - a];
    int testLabel = labels[labels.size() - a];
   
    //model1->set("threshold", 0.0);
   int label = -1; double confidence = 0.0;
    double x = (double)getTickCount();
   // int predictedLabel = model1->predict(testSample);
   model1->predict(testSample, label, confidence) ;
    x = ((double)getTickCount() - x)/getTickFrequency();
    cout << "Times passed in seconds: " << x << endl;

    myfile6 <<"data ke: "<<labels.size() - a;
    myfile6 <<";Prediksi "<<label<<";Label Aktual; "<<testLabel<<";confidence: "<<confidence<<";waktu deteksi: "<<x<<endl;
   // cout<<"confidence "<<confidence<<endl;
    cout<<"label Asli: "<< testLabel <<" Prediksi: "<<label<<" confidence: "<<confidence<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    cout <<"///////////////////////////////////////////////////////////////////////"<<endl;
    waitKey(10);
    imshow("LBPHRecognition", testSample);
    }
    myfile6.close();
        
}



void pengujian()
{
    //Mat img = imread("21.jpg");
    string fn_haar = "haarcascade_frontalface_default.xml";
    string filename = "datatest_robi.csv";


    vector<Mat> images;
    vector<int> labels;
    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    try {
        dbread(filename, images, labels);
    }
    catch (cv::Exception& e) {
        cerr << "Error opening file \"" << filename << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

    // int im_width = images[0].cols;
    //int im_height = images[0].rows;
    
    //cout<<"W"<<im_width<<endl;
    //cout<<"H"<<im_height<<endl;

    Ptr<FaceRecognizer> model2 = createEigenFaceRecognizer();
    model2->load("saveEigen.yml");

    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);

    int a;
    for(a=1;a<=145;a++)
    {
        cout<<"pengujian ke : "<<a<<" data ke "<<images.size() - a  <<endl;

        Mat testSample = images[images.size() -a];
        int testLabel = labels[labels.size() - a];
        //imshow("gaga",testSample);
       // Mat gray = testSample.clone();
       
       //  cout<<images[images.size() -a]<<endl;
           
      // images.pop_back();
       // labels.pop_back();
      
      // Mat gray;
      // cvtColor(testSample, gray, CV_BGR2GRAY);
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(testSample, faces);

    for (int i = 0; i < faces.size(); i++) 

        {
           Rect face_i = faces[i];
            // Crop the face from the image. So simple with OpenCV C++:
            Mat face = testSample(face_i);
           // imshow("face_crop", face); 
            Mat face_resized;
            cv::resize(face, face_resized, Size(111,154), 1.0, 1.0, INTER_CUBIC);
            // Now perform the prediction, see how easy that is:
            //int prediction = model->predict(face_resized);

            int label = -1; double confidence = 0;
            model2->predict(face_resized, label, confidence) ;
          cout<<"confidence "<<confidence<<endl;
            rectangle(testSample, face_i, CV_RGB(0, 255, 0), 1);
            // Create the text we will annotate the box with:
            string box_text;
           box_text = format("Prediction = ");
            // Get stringname
            if (label >= 0 && label <= 6)
            {
                box_text.append(g_listname_t[label]);
            }
            else box_text.append("Unknown");

            cout<<"label :"<< testLabel <<" Prediksi:"<<label<<endl;
            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
           putText(testSample, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);

       
        }
         imshow("face_recognizer", testSample);
    char key = (char)waitKey(200);            
   
    }
     //labels.pop_back();
        
       

}


