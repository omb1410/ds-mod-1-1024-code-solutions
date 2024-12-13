import pandas as pd 
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor 
from sklearn.metrics import mean_squared_log_error


def load_data(file_path):
    return pd.read_csv(file_path)

def drop_columns(df, cols_to_drop):
    df = df.drop(columns = cols_to_drop, axis=1)
    return df

def clean_data(df, train=True):
    si_num = SimpleImputer(strategy='mean')
    si_cat = SimpleImputer(strategy='most_frequent')

    if train:
        si_num.fit(df.select_dtypes(exclude='object'))
        si_cat.fit(df.select_dtypes(include='object'))

        with open("si_num.pickle", "wb") as f:
            pickle.dump(si_num, f)
        with open("si_cat.pickle", "wb") as f: 
            pickle.dump(si_cat, f)

    else:
        with open("si_num.pickle", "rb") as f:
            si_num = pickle.load(f)
        with open("si_cat.pickle", "rb") as f:
            si_cat = pickle.load(f)

    num_data = si_num.transform(df.select_dtypes(exclude='object'))
    cat_data = si_cat.transform(df.select_dtypes(include='object'))

    num_cols = df.select_dtypes(exclude='object').columns 
    cat_cols = df.select_dtypes(include='object').columns

    for i in range(len(num_cols)): 
        df[num_cols[i]] = num_data[:, i]

    for i in range(len(cat_cols)):
        df[cat_cols[i]] = cat_data[:, i]

    return df

def encode_data(df, train=True): 
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    if train: 
        categorical_cols = df.select_dtypes(include='object').columns
        encoded = ohe.fit_transform(df[categorical_cols]) 
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categorical_cols)) 
        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)

        with open("encoders.pickle", "wb") as f: 
            pickle.dump(ohe, f) 

    else: 
        with open("encoders.pickle", "rb") as f: 
            ohe = pickle.load(f) 

        categorical_cols = df.select_dtypes(include='object').columns
        encoded = ohe.transform(df[categorical_cols]) 
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categorical_cols)) 
        df = df.drop(columns=categorical_cols) 
        df = pd.concat([df, encoded_df], axis=1) 
    
    return df
            
def split_data(features_col, target_column):
    Xtrain, Xval, ytrain, yval = train_test_split(features_col, target_column, test_size=0.2, random_state=42)
    
    with open("Xtrain.pickle", "wb") as f:
       pickle.dump(Xtrain, f)
    with open("Xtest.pickle", "wb") as f:
       pickle.dump(Xval, f)
    with open("ytrain.pickle", "wb") as f:
        pickle.dump(ytrain, f)
    with open("ytest.pickle", "wb") as f:
        pickle.dump(yval, f)
    
    return Xtrain, Xval, ytrain, yval

def train_xgb_model(Xtrain, ytrain):
    xgb = XGBRegressor()
    xgb.fit(Xtrain, ytrain)
    with open("xgb.pickle", "wb") as f:
        pickle.dump(xgb, f)

    return xgb

def load_xgb_model():
    with open("xgb.pickle", "rb") as f: 
        model = pickle.load(f)
    
    return model

def make_preds(model, X):
    return model.predict(X) 

def evaluate_model(model, Xtest, ytest): 
    y_pred_val = model.predict(Xtest) 
    rmsle = np.sqrt(mean_squared_log_error(ytest, y_pred_val)) 
    return rmsle

def load_split_data(): 
    with open("Xtrain.pickle", "rb") as f: 
        Xtrain = pickle.load(f) 
    with open("Xtest.pickle", 'rb') as f: 
        Xtest = pickle.load(f) 
    with open("ytrain.pickle", 'rb') as f: 
        ytrain = pickle.load(f) 
    with open("ytest.pickle", 'rb') as f: 
        ytest = pickle.load(f) 

    return Xtrain, Xtest, ytrain, ytest




        

