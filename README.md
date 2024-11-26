### **Laptop Price Predictor**

#### **Environment Setup**
1. Install **virtualenv** and create a virtual environment:
   ```bash
   pip install virtualenv
   virtualenv [environment_name]
   ```
   Activate and install necessary libraries:
   - `jupyter`, `numpy`, `pandas`, and `sklearn`.

#### **Data Preparation**
1. **Import the Dataset**:
   - Load the dataset using pandas:  
     ```python
     data = pd.read_csv('data.csv', encoding='latin-1')
     ```
   - Preview and check dimensions:
     ```python
     data.head(), data.shape
     ```

2. **Handle Missing Values**:
   - Check for null values:
     ```python
     data.isnull().sum(), data.info()
     ```

3. **Data Conversion**:
   - Convert non-numerical features (e.g., `Ram`, `Weight`) into numerical types:
     ```python
     data['Ram'] = data['Ram'].str.replace('GB', '').astype('int32')
     data['Weight'] = data['Weight'].str.replace('kg', '').astype('float32')
     ```

4. **One-Hot Encoding**:
   - Transform categorical columns into numerical features:
     ```python
     data = pd.get_dummies(data)
     ```

5. **Feature Selection**:
   - Drop unnecessary columns:
     ```python
     data = data.drop(columns=['laptop_ID', 'Inches', 'Product', 'ScreenResolution', 'Cpu', 'Gpu'])
     ```
   - Select features (`X`) and target (`y`):
     ```python
     X = data.drop('Price_euros', axis=1)
     y = data['Price_euros']
     ```

#### **Model Training**
1. **Train-Test Split**:
   - Divide data into training and testing sets:
     ```python
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
     ```

2. **Train Models**:
   - Train various algorithms and evaluate accuracy:
     ```python
     from sklearn.linear_model import LinearRegression, Lasso
     from sklearn.tree import DecisionTreeRegressor
     from sklearn.ensemble import RandomForestRegressor

     def model_acc(model):
         model.fit(X_train, y_train)
         print(f"{model} --> {model.score(X_test, y_test)}")
     ```

   - Models used:
     - Linear Regression
     - Lasso Regression
     - Decision Tree Regressor
     - Random Forest Regressor

#### **Hyperparameter Tuning**
1. Use `GridSearchCV` to optimize the Random Forest model:
   ```python
   from sklearn.model_selection import GridSearchCV

   parameters = {'n_estimators': [10, 50, 100],
                 'criterion': ['squared_error', 'absolute_error', 'poisson']}
   grid_obj = GridSearchCV(estimator=RandomForestRegressor(), param_grid=parameters)
   grid_fit = grid_obj.fit(X_train, y_train)
   best_model = grid_fit.best_estimator_
   ```

#### **Save the Model**
- Save the trained model using `pickle`:
  ```python
  import pickle
  with open('predictor.pickle', 'wb') as file:
      pickle.dump(best_model, file)
  ```

### **Key Features**
- Comprehensive data preprocessing (missing values, type conversion, one-hot encoding).
- Model evaluation using multiple algorithms.
- Hyperparameter tuning with `GridSearchCV` for optimal performance.
- Model persistence using `pickle` for later use.
