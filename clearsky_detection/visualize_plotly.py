

import itertools
import numpy as np

# from bokeh.plotting import figure, show
# from bokeh.palettes import Category10_10, Dark2_8
# from bokeh.models import Range1d
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import figure_factory as ff

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer(object):

    def __init__(self, offline=True):
        if offline:
            init_notebook_mode(True)
        self.layout = {}
        self.data = []

    def add_line_ser(self, ser, label=None):
        if label is None:
            label = ser.name
        self.add_line(ser.index, ser.values, label)

    def add_circle_ser(self, ser, label=None):
        if label is None:
            label = ser.name
        self.add_circle(ser.index, ser.values, label)

    def add_line(self, x, y, label):
        self.data.append(go.Scatter(x=x, y=y, name=label))

    def add_circle(self, x, y, label):
        self.data.append(go.Scatter(x=x, y=y, name=label, mode='markers', marker={'size': 8}))

    def show(self):
        fig ={'data': self.data, 'layout': self.layout}
        # iplot(self.data, self.layout)
        iplot(fig)

    def add_date_slider(self):
        # self.layout['title'] = 'A plot'
        self.layout['xaxis'] = \
            {'rangeselector':
                {'buttons':
                    [{'count': 7, 'label': '1W', 'step': 'day', 'stepmode': 'forward'},
                     {'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'forward'},
                     {'step': 'all'}]
                },
            'rangeslider': {},
            'type': 'date',
            }

    def plot_confusion_matrix(self, mat, labels=None, normalize=True):
        if normalize:
            mat = np.round(mat.astype('float') / mat.sum(axis=1)[:, np.newaxis], 4)
        layout = {}
        layout['xaxis'] = {'title': 'Predicted label'}
        layout['yaxis'] = {'title': 'True label'}
        fig = ff.create_annotated_heatmap(mat, x=labels, y=labels, colorscale='Blues', showscale=True)
        # fig = ff.create_annotated_heatmap(mat, x=labels, y=labels)
        fig.layout.update({'width': 500})
        fig.layout.xaxis.update({'title': 'predicted label'})
        fig.layout.yaxis.update({'title': 'true label'})
        iplot(fig)

    def plot_corr_matrix(self, mat, labels):
        # fig = ff.create_annotated_heatmap(mat, x=labels, y=labels, colorscale='Blues', showscale=True)
        # fig.layout.update({'width': 500})
        # plot(fig)
        mask = np.zeros_like(mat, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(20, 10))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(mat, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=labels, yticklabels=labels)
        plt.yticks(rotation=0, fontsize=20)
        plt.xticks(rotation=90, fontsize=20)
        plt.show()

    def add_bar(self, x, y):
        self.data.append(go.Bar(x=x, y=y))

    def add_bar_ser(self, ser):
        self.add_bar(ser.index, ser.values)

    def plot_confusion_matrix2(self, cm, classes,
                              normalize=True,
                              title='Confusion matrix',
                              cmap=matplotlib.cm.get_cmap('Blues')):
        if normalize:
            cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 4)

        fig, ax = plt.subplots(figsize=(3,3))
        p = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title)
        fig.colorbar(p, fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        fig.tight_layout()
        return ax


def plot_ts_slider_highligther(df, meas='GHI', model='Clearsky GHI pvlib',
                               prob='prob', slider_lo=0.0, slider_hi=1, slider_num=21):
    trace_model = go.Scatter(x=df.index, y=df[model], name=model)
    trace_meas = go.Scatter(x=df.index, y=df[meas], name=meas)
    marker_dict = {'color': df[prob], 'colorscale': 'Jet', 'showscale': True, 'size': 8}
    trace_prob = go.Scatter(x=df.index, y=df[meas], mode='markers', marker=marker_dict,
                            text='P_clear=' + np.round(df[prob], 4).astype(str))
    data = [trace_meas, trace_model, trace_prob]

    for i in np.linspace(slider_lo, slider_hi, num=slider_num):
        trace = go.Scatter(x=df[df[prob] >= i].index, y=df[df[prob] >= i][meas], hoverinfo='none',
                           mode='markers', marker={'size': 12, 'symbol': 'circle-open', 'color': 'black'})
        data.append(trace)

    sliders = dict(
        # GENERAL
        steps = [],
        # currentvalue = dict(font=dict(size = 16), xanchor="left")),

        # PLACEMENT
        # x = 0.15,
        # y = -100,
        # len = 0.85,
        # pad = dict(t = 1, b = 0),
        # yanchor = "bottom",
        # xanchor = "left",
    )

    for ii, i in enumerate(np.linspace(slider_lo, slider_hi, num=slider_num)):
        step = dict(
            method = "restyle",
            label = str(i),
            value = str(i),
            args = ["visible", [False] * (slider_num + 3)], # Sets all to false
        )

        step['args'][1][0] = True # Main trace
        step['args'][1][1] = True # Main trace
        step['args'][1][2] = True # Main trace
        step['args'][1][3 + ii] = True # Selected trace through slider
        sliders['steps'].append(step)

    layout = dict(
        # title = chart_filename,
        showlegend = False,
        autosize = True,
        font = dict(size = 12),
        # margin = dict(t = 80, l = 50, b = 50, r = 50, pad = 5),
        # showlegend = True,
        sliders = [sliders]
    )

    figure = dict(data=data, layout=layout)
    iplot(figure)

def plot_ts_probas(df, meas='GHI', model='Clearsky GHI pvlib',
                               prob='prob', slider_lo=0.0, slider_hi=1, slider_num=21):
    trace_model = go.Scatter(x=df.index, y=df[model], name=model)
    trace_meas = go.Scatter(x=df.index, y=df[meas], name=meas)
    marker_dict = {'color': df[prob], 'colorscale': 'Jet', 'showscale': True, 'size': 8}
    trace_prob = go.Scatter(x=df.index, y=df[meas], mode='markers', marker=marker_dict,
                            text='P_clear=' + np.round(df[prob], 4).astype(str))
    data = [trace_meas, trace_model, trace_prob]

    figure = dict(data=data)
    iplot(figure)

def plot_confusion_matrix(mat, labels=None, normalize=True):
        if normalize:
            mat = np.round(mat.astype('float') / mat.sum(axis=1)[:, np.newaxis], 4)
        layout = {}
        layout['xaxis'] = {'title': 'Predicted label'}
        layout['yaxis'] = {'title': 'True label'}
        fig = ff.create_annotated_heatmap(mat, x=labels, y=labels, colorscale='Blues', showscale=True)
        # fig = ff.create_annotated_heatmap(mat, x=labels, y=labels)
        fig.layout.update({'width': 500})
        fig.layout.xaxis.update({'title': 'predicted label'})
        fig.layout.yaxis.update({'title': 'true label'})
        iplot(fig)

def plot_confusion_matrix2(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=matplotlib.cm.get_cmap('Blues')):
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 4)

    fig, ax = plt.subplots(figsize=(6, 6))
    p = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(p, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    return ax