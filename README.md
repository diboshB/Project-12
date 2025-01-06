# Project-12
Capstone_Project - Regression - Fuel_Efficiency_Prediction


1. Introduction
The goal of this project is to develop a machine learning model that predicts
the fuel consumption of vehicles based on various features, such as the
vehicle's mass, engine capacity, fuel type, emissions, energy consumption,
and other relevant characteristics. This model can help manufacturers and
researchers understand the factors influencing fuel efficiency and optimize
vehicle designs for better performance. By leveraging historical data from a
large dataset, this project aims to predict fuel consumption, which can be a
valuable tool in improving vehicle fuel efficiency, reducing emissions, and
enhancing overall environmental sustainability.
2. Data Collection
The dataset used for this project consists of 1 million records, which
includes various features related to automobiles. Each record represents a
vehicle with different specifications, including its mass, fuel type, energy
consumption, and fuel consumption values. The data includes the following
attributes:
• Mass (m): The weight of the vehicle (in kilograms).
• Engine capacity (ec): The engine displacement of the vehicle (in
cubic centimeters).
• Fuel type (Ft): The type of fuel used by the vehicle (e.g., petrol,
electric, diesel, etc.).
• Energy consumption (z): The energy consumption of the vehicle (in
Wh/km).
• Fuel consumption: The target variable representing the fuel
consumption of the vehicle (in liters per 100 kilometers).
• Other features: Several other characteristics such as engine power,
emissions, electric range (for electric vehicles), and more.
The dataset is split into three portions:
• Training set: The first 700,000 records used to train the model.
• Evaluation set: The next 200,000 records used to evaluate the
model's performance.
• Production set: The remaining records are used for future predictions
once the model is deployed in production.
3. Data Preprocessing
To make the data suitable for model training, several preprocessing steps
were applied:
3.1 Handling Missing Values
The dataset was checked for missing values in various columns. Missing
values in the numerical columns (e.g., mass, engine capacity, fuel
consumption) were imputed using the median of the respective columns.
For categorical features like fuel type and manufacturer, missing values were
filled using the mode (the most common category). This ensures that the
data is complete and ready for machine learning.
3.2 Text Cleaning
Some features, particularly categorical ones (e.g., fuel type), might have
extraneous characters or spaces. These were removed by stripping
leading/trailing spaces from column names and ensuring consistency in
feature naming.
3.3 Feature Engineering
Several columns with too many missing values or irrelevant information
were dropped to reduce complexity. Columns like z (Wh/km) and Erwltp (g/km)
had too many missing values, so they were discarded from the dataset.
Additionally, features that don't contribute significantly to predicting fuel
consumption, such as the r column, were also dropped.
3.4 One-Hot Encoding
Categorical columns like fuel type and manufacturer were one-hot
encoded. One-hot encoding converts categorical features into a format that
can be fed into machine learning algorithms by creating binary variables for
each possible category. This step ensures that the machine learning model
can understand and use categorical data effectively.
3.5 Feature and Target Separation
The dataset was split into features (X) and the target variable (y), where the
target is fuel consumption, and the features include other attributes like
mass, engine capacity, fuel type, and energy consumption.
4. Model Building
The primary objective of this project is to predict fuel consumption using a
machine learning model. A Random Forest Regressor was chosen for this
task, as it is a powerful ensemble learning method that performs well with
both categorical and numerical data.
4.1 Model Training
The Random Forest Regressor was trained using the training set of
700,000 records. Random Forest works by creating multiple decision trees,
each trained on a random subset of the data. The predictions from these
trees are aggregated to give the final output, which in this case is the
predicted fuel consumption of the vehicle.
4.2 Model Evaluation
After training the model, it was evaluated on the evaluation set of 200,000
records to determine its performance. The following evaluation metrics were
used:
• RMSE (Root Mean Squared Error): This metric measures the
average magnitude of errors in the predictions, giving more weight to
larger errors.
• R² (Coefficient of Determination): This value indicates how well
the model explains the variance in fuel consumption. A high R²
indicates that the model performs well.
• MAE (Mean Absolute Error): This metric measures the average
absolute errors in the predictions, indicating how far the model's
predictions are from the true values.
• MSLE (Mean Squared Logarithmic Error): This metric is useful
when the target variable has a wide range or exponential growth. It
penalizes large differences in predictions, especially when they are
large values.
• Explained Variance Score: This metric indicates the proportion of
the variance in fuel consumption that is explained by the model.
5. Model Evaluation Results
The model performed exceptionally well on the evaluation set, producing the
following key results:
• RMSE (Root Mean Squared Error): 0.1893
This indicates that, on average, the model's predictions deviate by
around 0.1893 units of fuel consumption, which is very low.
• R² (Coefficient of Determination): 0.989
This indicates that 98.9% of the variance in fuel consumption can be
explained by the model, which is an excellent performance.
• MAE (Mean Absolute Error): 0.0496
The average error is very small, indicating that the model's
predictions are very close to the actual values.
• MSLE (Mean Squared Logarithmic Error): 0.0011
This very low value indicates that the model performs well even when
predicting fuel consumption values with a wide range.
• Explained Variance Score: 0.989
This confirms that the model captures almost all of the variability in
fuel consumption.
These results demonstrate that the Random Forest Regressor model is
highly accurate and effective in predicting fuel consumption based on the
features in the dataset.
6. Results & Discussion
6.1 Model Insights
The model successfully predicts fuel consumption with a high degree of
accuracy. The high R² value suggests that the model is able to capture the
relationship between various features (such as vehicle mass, engine
capacity, fuel type, etc.) and fuel consumption. The relatively low RMSE and
MAE indicate that the model's predictions are quite close to the actual
values.
6.2 Feature Importance
One of the advantages of using a Random Forest model is that it allows us to
assess the importance of each feature in predicting the target variable.
Features such as vehicle mass, engine capacity, and fuel type were
found to be the most important predictors of fuel consumption. This makes
sense, as larger, more powerful vehicles tend to consume more fuel. The
energy consumption (z) feature was also important, especially for electric
vehicles, highlighting the significance of energy efficiency in fuel
consumption prediction.
6.3 Limitations
While the model performs well overall, it is important to acknowledge a few
limitations:
• The dataset may not fully account for all factors influencing fuel
consumption, such as driving behavior, road conditions, or
maintenance status of the vehicles.
• There may be potential biases in the data that affect the model’s
performance, such as an overrepresentation of certain vehicle types or
fuel types.
6.4 Future Improvements
To further improve the model, additional features such as driving conditions,
vehicle age, and detailed maintenance history could be included. Moreover,
employing more advanced algorithms like Gradient Boosting or XGBoost
might help achieve even better predictive performance.
7. Conclusion
This project successfully developed a machine learning model to predict fuel
consumption based on vehicle characteristics. The Random Forest Regressor
model achieved high accuracy, with an R² of 0.989, indicating that it
effectively captures the factors influencing fuel consumption. The evaluation
metrics, including RMSE, MAE, and MSLE, all demonstrate the model’s
excellent performance. By identifying key features contributing to fuel
consumption, this model can provide valuable insights for manufacturers and
researchers working to optimize vehicle fuel efficiency and reduce
environmental impact.
