#include <chrono> // for measuring time
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// Function to read in the Titanic dataset
vector<vector<double>> read_csv(string filename) {
  vector<vector<double>> data;
  ifstream file(filename);

  if (!file.is_open()) {
    cout << "Error opening file!" << endl;
    return data;
  }

  string line;
  while (getline(file, line)) {
    vector<double> row;
    stringstream ss(line);

    string value;
    while (getline(ss, value, ',')) {
      // Check if the value can be converted to a double
      bool is_double = true;
      for (char c : value) {
        if (!isdigit(c) && c != '.') {
          is_double = false;
          break;
        }
      }
      if (is_double) {
        row.push_back(stod(value));
      } else {
        // Handle error case
        cout << "Error: Invalid value encountered in CSV file" << endl;
        data.clear();
        return data;
      }
    }

    data.push_back(row);
  }

  file.close();
  return data;
}

// Function to perform logistic regression on the training data
vector<double> logistic_regression(vector<vector<double>> train_data,
                                   double learning_rate, int num_iterations) {
  int n = train_data.size();    // number of samples
  int d = train_data[0].size(); // number of features (including the intercept)

  // Initialize weights to zero
  vector<double> w(d, 0);

  // Perform gradient descent
  for (int i = 0; i < num_iterations; i++) {
    // Compute the predictions using the current weights
    vector<double> predictions(n);
    for (int j = 0; j < n; j++) {
      double z = 0;
      for (int k = 0; k < d; k++) {
        z += w[k] * train_data[j][k];
      }
      predictions[j] = 1.0 / (1.0 + exp(-z));
    }

    // Compute the gradients
    vector<double> gradients(d, 0);
    for (int k = 0; k < d; k++) {
      double gradient = 0;
      for (int j = 0; j < n; j++) {
        gradient += (train_data[j][d - 1] - predictions[j]) * train_data[j][k];
      }
      gradients[k] = gradient;
    }

    // Update the weights
    for (int k = 0; k < d; k++) {
      w[k] += learning_rate * gradients[k];
    }
  }

  return w;
}
int main() {
  // Step 1: read in Titanic dataset
  vector<vector<double>> data = read_csv("titanic.csv");

  // Step 2: prepare the data for logistic regression
  int n = 800;
  int d = 2;
  vector<vector<double>> train_data(n, vector<double>(d));
  for (int i = 0; i < n; i++) {
    train_data[i][0] = data[i][3]; // sex
    train_data[i][1] = data[i][0]; // intercept
  }

  // Step 3: perform logistic regression on sex to predict survived
  double learning_rate = 0.1;
  int num_iterations = 1000;
  vector<double> coeffs =
      logistic_regression(train_data, learning_rate, num_iterations);

  // Step 4: output coefficients
  cout << "Coefficients: " << coeffs[0] << ", " << coeffs[1] << endl;

  return 0;
}
