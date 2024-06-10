# bigmart_sales_prediction
Bigmart Sales Prediction Using ML

# BigMart Sales Prediction

This project aims to predict sales for a retail dataset using various machine learning models. The dataset contains information about different items sold at various stores and their respective sales figures.

## Dataset Description

The dataset contains the following columns with 8523 rows:

- `Item_Identifier`: Unique identifier for each item
- `Item_Weight`: Weight of the item
- `Item_Fat_Content`: Whether the item is low fat or regular
- `Item_Visibility`: The percentage of total display area of all products in a store allocated to the particular product
- `Item_Type`: The category to which the item belongs
- `Item_MRP`: Maximum Retail Price (list price) of the item
- `Outlet_Identifier`: Unique identifier for the outlet
- `Outlet_Establishment_Year`: The year in which the outlet was established
- `Outlet_Size`: The size of the outlet in terms of ground area covered
- `Outlet_Location_Type`: The type of city in which the outlet is located
- `Outlet_Type`: Whether the outlet is a grocery store or some sort of supermarket
- `Item_Outlet_Sales`: Sales of the particular item in the particular outlet

## Data Preprocessing

1. **Basic Information:**
   - Checked for null values, dataset shape, and descriptive statistics using `info()`, `head()`, and `describe()`. This step ensures we understand the structure and contents of the dataset.

2. **Feature Extraction and Transformation:**
   - **Item Identifier:**
     - Extracted the first letter from `Item_Identifier` to categorize items into "F" (Food), "N" (Non-food), and "D" (Drinks).
     - Mapped these categories to numerical values:
       ```python
       df['Item_Identifier'] = df['Item_Identifier'].map({"F": 0, "N": 1, "D": 2})
       ```
   - **Item Fat Content:**
     - Consolidated `Item_Fat_Content` values to ensure consistency:
       ```python
       df['Item_Fat_Content'] = df['Item_Fat_Content'].map({"Low Fat": 0, "LF": 0, "low fat": 0, "reg": 1, "Regular": 1})
       ```
   - **Outlet Identifier:**
     - Extracted numerical values from `Outlet_Identifier`:
       ```python
       df['Outlet_Identifier'] = df["Outlet_Identifier"].apply(lambda x: int(x[3:]))
       ```
   - **Outlet Location Type:**
     - Mapped `Outlet_Location_Type` to numerical values:
       ```python
       df["Outlet_Location_Type"] = df["Outlet_Location_Type"].map({"Tier 1": 0, "Tier 2": 1, "Tier 3": 2})
       ```

3. **Handling Missing Values:**
   - **Outlet Size:**
     - Filled missing values in `Outlet_Size` and converted to numerical values:
       ```python
       df["Outlet_Size"].fillna("Medium", inplace=True)
       df["Outlet_Size"] = df["Outlet_Size"].map({"Small": 0, "Medium": 1, "High": 2})
       ```
   - **Item Weight:**
     - Filled missing values in `Item_Weight` by calculating the mean weight of each `Item_Type`:
       ```python
       items_mean = dict(df.groupby("Item_Type")["Item_Weight"].mean())
       def weight(x):
           if np.isnan(x[0]):
               return round(items_mean[x[1]], 3)
           else:
               return x[0]
       df["Item_Weight"] = df[["Item_Weight", "Item_Type"]].apply(weight, axis=1)
       ```

4. **One-Hot Encoding:**
   - Applied one-hot encoding to `Outlet_Type` and `Item_Type` to convert categorical variables into numerical format:
     ```python
     df = df.join(pd.get_dummies(df["Outlet_Type"], drop_first=True, dtype=int))
     df.drop("Outlet_Type", inplace=True, axis=1)
     df = df.join(pd.get_dummies(df["Item_Type"], drop_first=True, dtype=int))
     df.drop("Item_Type", axis=1, inplace=True)
     ```

5. **Correlation Analysis:**
   - Plotted a heatmap to check the correlation between columns. This step helps in understanding the relationships and multicollinearity between features.

## Model Training and Evaluation

1. **Train-Test Split:**
   - Split the dataset into training and testing sets to evaluate model performance on unseen data.

2. **Models Used:**
   - **Linear Regression:** A simple and interpretable model.
   - **XGBRegressor:** An advanced and powerful model that often performs well on tabular data.

3. **Evaluation Metrics:**
   - **Mean Squared Error (MSE):** Measures the average of the squares of the errors.
   - **R² Score:** Indicates how well the model explains the variance of the target variable.
   - **Root Mean Squared Error (RMSE):** The square root of the average of squared differences between prediction and actual observation.

4. **Results:**
   - **Linear Regression:** R² Score = 0.5788
   - **XGBRegressor:** R² Score = 0.5228

   The Linear Regression model performed better than the XGBRegressor. The lower performance of XGBRegressor could be due to overfitting or the need for hyperparameter tuning. Feature selection and regularization might improve its performance further.

## Conclusion

This project highlights the importance of data preprocessing, feature engineering, and model evaluation. While Linear Regression provided a decent R² score, further optimization and exploration of different models could enhance prediction accuracy.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/riitk/bigmart_sales_prediction.git
   cd bigmart_sales_prediction
