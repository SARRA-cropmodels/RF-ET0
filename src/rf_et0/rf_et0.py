import numpy as np
import pandas as pd
from pcse.db import NASAPowerWeatherDataProvider
from pcse.util import reference_ET
import xgboost as xgb

def train_RF_ET0_model(lat,long,max_train_date,training_variables,elev=np.nan):
    
    angstA = 0.29
    angstB = 0.49

    # getting weather data
    print("> Retrieving NASA Power data")
    weatherdata = NASAPowerWeatherDataProvider(longitude=long, latitude=lat)
    print(weatherdata)
    df_NASAPower = pd.DataFrame(weatherdata.export())
    df_NASAPower = df_NASAPower[df_NASAPower["DAY"]<max_train_date].reset_index(drop=True)
    df_NASAPower["DOY"] = df_NASAPower.apply(lambda x: x["DAY"].timetuple().tm_yday, axis=1)
    df_NASAPower["ET0"] = df_NASAPower["ET0"] * 10 #conversion cm to mm
    df_NASAPower["RAIN"] = df_NASAPower["RAIN"] * 10 #conversion cm to mm

    first_NASAPower_date = df_NASAPower.loc[0,"DAY"]

    print("Model training will be performed using data ranging from",first_NASAPower_date,"to",max_train_date,"\n")

    if not np.isnan(elev):
        print("> Exact elevation provided, calculating corrected ET0...")
        df_NASAPower["ELEV"] = elev
        df_NASAPower["ET0"] = df_NASAPower.apply(lambda x: reference_ET(x["DAY"], x["LAT"], x["ELEV"], x["TMIN"], x["TMAX"], x["IRRAD"], x["VAP"], x["WIND"], angstA, angstB)[2], axis=1)

    # training model
    print("> Calculating cross-validation metrics...")
    X, y = df_NASAPower[training_variables],df_NASAPower[["ET0"]]
    data_dmatrix = xgb.DMatrix(data=X,label=y)

    num_boost_rounds = 1000
    objective = "reg:squarederror" 
    colsample_bytree = 0.3 # default
    learning_rate = 0.1 # default
    max_depth = 5 # default
    alpha = 10 # default
    early_stopping_rounds = 10 # default

    # cross validation
    params = {"objective":objective,
        'colsample_bytree': colsample_bytree,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'alpha': alpha}

    cv_results = xgb.cv(dtrain=data_dmatrix,
        params=params,
        nfold=10,
        num_boost_round=num_boost_rounds,
        early_stopping_rounds=early_stopping_rounds,
        metrics="rmse",
        as_pandas=True,
        # seed=123,
        verbose_eval=None)

    # proper training on all data
    print("> Training RF model...")
    xg_reg = xgb.XGBRegressor(objective = objective,
        colsample_bytree = colsample_bytree,
        learning_rate = learning_rate,
        max_depth = max_depth,
        alpha = alpha,
        n_estimators = num_boost_rounds, # Equivalent to number of boosting rounds
        verbosity=1,
        )

    xg_reg.fit(X, y)

    print("> Done !")

    return df_NASAPower, cv_results, xg_reg