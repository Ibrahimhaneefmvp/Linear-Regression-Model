# California Housing Price Prediction Using Linear Regression
Project Overview
This project is part of my machine learning internship (Week 3). The objective was to create a Linear Regression model using the California Housing Dataset to predict house prices based on three key features:

Median Income
House Age
Average Rooms
Additionally, custom inputs can be provided to predict house prices based on these three features. The results of these predictions are visualized on a scatter plot alongside the regression line.

Dataset
The dataset used in this project is the California Housing Dataset, which contains information on various housing characteristics in California districts.

Features used:
MedInc (Median income in a block)
HouseAge (Average house age)
AveRooms (Average number of rooms per house)
Target:
MedianHouseValue (Median house value in a block)
Installation
Step 1: Clone the repository
bash
Copy code
git clone https://github.com/your-repository/california-housing-price-prediction.git
cd california-housing-price-prediction
Step 2: Install dependencies
Ensure that you have Python 3.x and pip installed. Then, install the required libraries:

bash
Copy code
pip install -r requirements.txt
The required libraries are:

numpy
pandas
matplotlib
seaborn
scikit-learn
Step 3: Running the Jupyter Notebook
You can launch the Jupyter Notebook to explore the code and visualize results.

bash
Copy code
jupyter notebook california_housing_regression.ipynb
## Usage
Load the dataset, preprocess it, and split it into training and testing sets.
Train a Linear Regression model using the features mentioned above.
Provide custom inputs (Median Income, House Age, and Average Rooms) to predict the house price.
Visualize the regression line, actual values, and custom predictions on a scatter plot.
# Code Walkthrough
## Data Loading and Preprocessing

Load the California Housing Dataset and filter the necessary columns (MedInc, HouseAge, AveRooms, and MedHouseVal).
Split the data into training and testing sets (80% train, 20% test).
## Model Training

Train a Linear Regression model using the training set.
Evaluate the model on the test set using metrics like Mean Squared Error (MSE) and R-squared (R²).
## Custom Prediction Input

Allow users to input values for the three features (Median Income, House Age, and Average Rooms) to predict the house price.
Visualize the prediction result on the same plot as the regression line.
## Visualization

Create a scatter plot with the actual values and regression line.
Overlay the custom predictions on the plot to compare with actual data points.
## Features
Predict house prices using a Linear Regression model.
Custom inputs allow for dynamic prediction and visualization of house prices based on real-time user inputs.
Visualize results with a regression line and custom predictions in an intuitive plot.
## Model Performance
Evaluated the model using Mean Squared Error (MSE) and R² score to determine the accuracy of predictions.
Visual results clearly demonstrate the relationship between the features and the target variable.
## Results
The trained model was able to provide reasonable predictions of house prices based on the three selected features, with a visual comparison of real vs predicted data.

## Future Improvements
Experiment with more advanced models like Polynomial Regression or Random Forest to improve prediction accuracy.
Include more features from the dataset, such as population, latitude, and longitude.
