"""This module will help us plot a visualizations."""

# import data anlysis packages:
import numpy as np

# import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_train_vs_val_loss(history=None, a=None):
    """
    plots traning vs validation losses.
    Argument:
        history: model history 
    Outputs:
        plot for traning vs validation loss
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history['val_loss'], name="validation Loss"))
    fig.update_layout(
                      title= {'text': "Training vs validation loss ({})".format(a),
                              'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      xaxis_title="Epochs",
                      yaxis_title="Loss",
                      font=dict(family="Courier New, monospace", size=16, color='black'),
                     )

    fig.show();
