#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <numeric>
using namespace std;

void print_stats(vector<double> data) {
    sort(data.begin(), data.end());
    cout << "Mean: " << accumulate(data.begin(), data.end(), 0.0) / data.size() << endl;
    cout << "Median: " << data[data.size() / 2] << endl;
    cout << "Max: " << data.back() << endl;
    cout << "Min: " << data.front() << endl;
}

double covar(vector<double> x, vector<double> y) {
    double x_mean = accumulate(x.begin(), x.end(), 0.0) / x.size();
    double y_mean = accumulate(y.begin(), y.end(), 0.0) / y.size();

    double result = 0;
    for (int i = 0; i < x.size(); i++) {
        result += (x[i] - x_mean) * (y[i] - y_mean);
    }

    return result / (x.size() - 1);
}

double cor(vector<double> x, vector<double> y) {
    return covar(x, y) / sqrt(covar(x, x) * covar(y, y));
}

int main(int argc, char** argv){
    ifstream inFS; //input file stream
    string line;
    string rm_in, medv_in;
    const int MAX_LEN = 1000; 
    vector<double> rm(MAX_LEN);
    vector<double> medv(MAX_LEN);

    //try to open file
    cout<< "Opening file Boston.csv." << endl;

    inFS.open("Boston.csv");
    if(!inFS.is_open()){
        cout<< "Could not open file Boston.csv."<< endl;
        return 1; //1 indicates error

    }
    //can now use inFS stream like cin stream 
    //boston.csv should contain 2 doubles

    cout<< "Reading Line 1" << endl;
    getline(inFS, line);
    
    //echo heading
    cout<< "heading: " << line << endl;

    int numObservations = 0;
    while(inFS.good()){

        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, ',');

        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);
        
        numObservations++; 

    }

    rm.resize(numObservations);
    medv.resize(numObservations);

    cout<< "new length" <<rm.size() << endl;

    cout<< "Closing file Boston.csv." << endl;
    inFS.close(); // Done with file closing

    cout<< "\nStats for rm" << endl;
    print_stats(rm);

    cout<< "\nStats for medv" << endl;
    print_stats(medv);

    cout << "\n Covariance = " << covar(rm, medv) << endl;

    cout << "\n Correlation = " << cor(rm, medv) << endl;

    cout << "\n Program terminated." ;


    return 0;
}
