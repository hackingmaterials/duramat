

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
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(mat, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=labels, yticklabels=labels)
        plt.yticks(rotation=0, fontsize=6)
        plt.xticks(rotation=90, fontsize=6)
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
