

import itertools

from bokeh.plotting import figure, show
from bokeh.palettes import Category10_10, Dark2_8
from bokeh.models import Range1d

class Visualizer(object):

    def __init__(self, x_axis_type='datetime', height=500, width=1000, line_width=2, circle_size=8):
        self.height = height
        self.width = width
        self.line_width = line_width
        self.circle_size = circle_size
        # self.fig.legend.click_policy = 'hide'
        self.line_colors = itertools.cycle(Category10_10)
        self.line_curr_color = Category10_10[0]
        self.scatter_colors = itertools.cycle(Dark2_8)
        self.scatter_curr_color = Dark2_8[0]
        self.fig = figure(x_axis_type=x_axis_type, plot_width=self.width, plot_height=self.height,
                          tools='pan,wheel_zoom,box_zoom,reset,undo,save')

    def add_line_ser(self, ser, label=None):
        if label is None:
            label = ser.name
        self.add_line(ser.index, ser.values, label)

    def add_circle_ser(self, ser, label=None):
        if label is None:
            label = ser.name
        self.add_circle(ser.index, ser.values, label)

    def add_line(self, x, y, label, color=None, line_width=2):
        if color is None:
            color = next(self.line_colors)
        self.fig.line(x, y, legend=label, color=color, line_width=line_width)

    def add_circle(self, x, y, label, color=None, size=10):
        if color is None:
            color = next(self.scatter_colors)
        self.fig.circle(x, y, legend=label, color=color, size=size)

    def set_x_range(self, x0, x1):
        self.fig.x_range = Range1d(x0, x1)

    def show(self, legend_policy='hide'):
        self.fig.legend.click_policy = legend_policy
        show(self.fig)

    def get_fig(self):
        return self.fig
