import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Output, Input

dataset = pd.read_csv('data_1995_2019_f.csv', low_memory=False)
dataset = dataset[dataset.price != 'price']
data = dataset.drop(['Unnamed: 0','Elektryczny', 'Hybryda', 'CNG', 'page', 'version'], axis=1)
data[["price", "mileageFromOdometer","year"]] = data[["price", "mileageFromOdometer","year"]].astype(np.float32)
data[["year"]] = data[["year"]].round(0).astype(np.int64)
data1 = data.dropna()
data2 = data1.groupby(by=["brand", "model", "year"], dropna=False, as_index=False).mean()
data3 = data1.groupby(by=["brand", "model", "year"], dropna=False, as_index=False).agg({
                                                                                        'price': 'count'})
data2['count'] = data3['price']
data2 = data2.loc[data2['count'] > 50]
print(data2)
app = Dash(__name__)


app.layout = html.Div([
    html.H4('otomoto'),
    dcc.Dropdown(data2['brand'].unique(),id='dropdown',),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"),
    Input("dropdown","value"))
def update_line_chart(brand):

    df = data2.loc[(data2['brand'] == brand)]

    fig = px.line(df,
                  x="year", y="price", color='model')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)