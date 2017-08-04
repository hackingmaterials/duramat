

import itertools

# from bokeh.plotting import figure, show
# from bokeh.palettes import Category10_10, Dark2_8
# from bokeh.models import Range1d
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

class Visualizer(object):

    def __init__(self, offline=True):
        if offline:
            init_notebook_mode(True)
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
        self.data.append(go.Scatter(x=x, y=y, name=label, mode='markers'))

    def show(self):
        iplot(self.data)
