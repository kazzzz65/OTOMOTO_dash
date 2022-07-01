# This is a sample Python script.
import numpy as np
import pandas as pd
import plotly
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def preprocess(data):
    for i in range(0, len(data['model'].values)):
        if data['model'].values[i] == 'Seria' or data['model'].values[i] == 'Klasa' or data['model'].values[
            i] == 'Rover':
            data['model'].values[i] = data['version'].values[i]
            data['version'].values[i] = data[3].values[i]
        elif data['priceSpecification.price'].values[i] == 'undefined':
            data['priceSpecification.price'].values[i] = np.nan
        elif data['itemOffered.mileageFromOdometer.value'].values[i] == 'undefined':
            data['itemOffered.mileageFromOdometer.value'].values[i] = np.nan
    data = data.reset_index()
    data = data[["priceSpecification.price", "itemOffered.brand", "itemOffered.fuelType", "itemOffered" \
                                                                                          ".mileageFromOdometer.value",
                 "model", "version", "year", "page"]]

    data[["priceSpecification.price", "itemOffered.mileageFromOdometer.value"]] = data[
        ["priceSpecification.price", "itemOffered.mileageFromOdometer.value"]].astype(np.float32)
    data[["year", "page"]] = data[["year", "page"]].astype(np.float32)

    for i in range(0, len(data['version'].values)):
        if data['version'].values[i] == 'None':
            data['version'].values[i] = np.nan

    categorical_cols1 = ["itemOffered.fuelType"]
    categorical_cols = [cname for cname in data.columns if
                        data[cname].dtype == "object"]

    categorical_transformer = SimpleImputer(strategy='most_frequent')
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.float32)
    data[categorical_cols] = categorical_transformer.fit_transform(data[categorical_cols])
    data1 = data.copy().drop(categorical_cols1, axis=1)
    data_OH = pd.DataFrame(oh_encoder.fit_transform(data[categorical_cols1]))
    data_OH = data_OH.set_axis(oh_encoder.get_feature_names_out(), axis=1, inplace=False)
    data_OH.index = data.index
    data2 = pd.concat([data1, data_OH], axis=1)

    # Select numerical columns
    numerical_cols = [cname for cname in data2.columns if data2[cname].dtype in ['float32']]
    numerical_transformer = SimpleImputer(strategy='median')
    data2[numerical_cols] = numerical_transformer.fit_transform(data2[numerical_cols])
    data2 = pd.DataFrame(data2)

    data2.rename(
        columns={'priceSpecification.price': 'price', 'itemOffered.mileageFromOdometer.value': 'mileageFromOdometer',
                 'itemOffered.brand': 'brand', 'itemOffered.fuelType': 'fuelType'}, inplace=True)
    data2.rename(columns={'itemOffered.fuelType_Benzyna': 'Benzyn', 'itemOffered.fuelType_Benzyna+CNG': 'CNG',
                          'itemOffered.fuelType_Benzyna+LPG': 'LPG', 'itemOffered.fuelType_Diesel': 'Diesel',
                          'itemOffered.fuelType_Hybryda': 'Hybryda', 'itemOffered.fuelType_Elektryczny': 'Elektryczny'},
                 inplace=True)

    return data2


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = pd.read_csv('data_2020_2021_f.csv', low_memory=False)
    dataset = preprocess(dataset)

    data = dataset.drop(['Elektryczny', 'Hybryda', 'CNG', 'page'], axis=1)
    data1 = data.dropna()
    data2 = data1.groupby(by=["brand", "model", "version", "year"], dropna=False, as_index=False).mean()
    import plotly.express as px

    df = data2.loc[(data2['brand'] == 'Audi') & (data2['model'] == 'A4')]
    fig = px.line(df, x="year", y="price", color='version', )
    fig.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/






