from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
class Preprocess:
    def __init__(self, df):
        self.df = df

    def handle_outliers(self, columns, data):
        # Handle outliers
        Q1 = data[columns].quantile(0.25)
        Q3 = data[columns].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_indices = []
        for col in columns:
            outliers = data[
                (data[col] < lower_bound[col]) | (data[col] > upper_bound[col])].index
            outlier_indices.extend(outliers)
        data.drop(outlier_indices, inplace=True)
        return data

    def dropColumns(self,columns,  data):
        data.drop(columns, axis=1, inplace=True)
        return data

    def encode(self, columns):
        # Take copy of original X
        self.df = self.df.copy()
        # Declare dictionary
        for x in columns:
            # Create object of LabelEncoder
            le = LabelEncoder()
            # Learn encoder model on ele of X
            le.fit(self.df[x].values)
            # Implement encode on ele of X
            self.df[x] = le.transform(list(self.df[x].values))

    def scale_data(self, columns):
        scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])

    def CleanData(self, df):
        self.df.drop('customerID', axis=1, inplace=True)
        print(self.df.dtypes)
        self.df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        ################################# Constant Columns ###########################################
        print('######################## Unique Columns ####################################')
        unique_counts = self.df.nunique()
        constant_columns = unique_counts[unique_counts == 1].index
        if constant_columns.empty:
            print('There is no unique columns')
        else:
            print('Unique columns is: ', constant_columns)

        self.df.drop(constant_columns, axis=1, inplace=True)
        print('######################## The End ###########################################')

        ################################## Handle Nulls ############################################
        print('######################## Null values #####################################')
        print(self.df.isnull().sum())
        self.df.dropna(inplace=True)
        print(self.df.isnull().sum())
        print('######################## The End ##########################################')
        ################################## Handle Duplicate #########################################
        print('######################## Duplicate Rows ###################################')
        duplicate_rows = self.df.duplicated()
        print("                Befor\nNumber of duplicate rows:", duplicate_rows.sum())
        self.df.drop_duplicates(inplace=True)
        duplicate_rows = self.df.duplicated()
        print("                After\nNumber of duplicate rows:", duplicate_rows.sum())
        print('######################## The End ##########################################')

        ################################# Handle Outliers ############################################
        columns=['tenure', 'MonthlyCharges', 'TotalCharges']
        self.df = self.handle_outliers(columns, self.df)

        ################################## Encode Categorical Data ###################################
        for index, value in df['MultipleLines'].items():
            if value == 'No phone service':
                self.df.at[index, 'MultipleLines'] = 'No'

        for index, value in df['OnlineSecurity'].items():
            if value == 'No internet service':
                self.df.at[index, 'OnlineSecurity'] = 'No'

        for index, value in df['OnlineBackup'].items():
            if value == 'No internet service':
                self.df.at[index, 'OnlineBackup'] = 'No'

        for index, value in df['DeviceProtection'].items():
            if value == 'No internet service':
                self.df.at[index, 'DeviceProtection'] = 'No'

        for index, value in df['StreamingTV'].items():
            if value == 'No internet service':
                self.df.at[index, 'StreamingTV'] = 'No'

        for index, value in df['StreamingMovies'].items():
            if value == 'No internet service':
                self.df.at[index, 'StreamingMovies'] = 'No'

        print('######################## Encode Columns ####################################')
        listEn = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',	'OnlineBackup',	'DeviceProtection',	'TechSupport',	'StreamingTV',	'StreamingMovies',	'Contract',	'PaperlessBilling',	'PaymentMethod','Churn']
        print(self.df[listEn].head)
        self.encode(listEn)
        print('######################## The End ###########################################')

        ################################# Scale #####################################
        print('######################## Scale Columns ####################################')
        listScale = ['tenure', 'MonthlyCharges', 'TotalCharges']
        print(self.df[listScale].head)
        self.scale_data(listScale)
        print('######################## The End ###########################################')
        return self.df