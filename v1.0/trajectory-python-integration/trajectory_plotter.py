import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np

# matplotlib plotting parameters
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times'
mpl.rcParams['figure.titlesize'] = 'xx-large'
mpl.rcParams['axes.labelsize'] = 21
mpl.rcParams['axes.titlesize'] = 21
mpl.rcParams['xtick.labelsize'] = 21
mpl.rcParams['ytick.labelsize'] = 21

# подключаем пакеты для рендеринга русского текста в LaTeX
mpl.rc('text.latex', preamble=r'\usepackage{amsmath}')
mpl.rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
mpl.rc('text.latex', preamble=r"\usepackage[russian]{babel}")

class TrajectoryPlotter:
    def __init__(self, times, phase_points, figsize=(10, 7.5)):
        self.times = times
        self.phase_points = phase_points
        
        self.fig = plt.figure(figsize = figsize)
        self.ax = self.fig.add_subplot(111)
        
        self.xlim, self.ylim = None, None
        
    def set_locators(self, locators):
        self.ax.xaxis.set_major_locator(plt.MultipleLocator(locators['xaxis.major']))
        self.ax.xaxis.set_minor_locator(plt.MultipleLocator(locators['xaxis.minor']))
        self.ax.yaxis.set_major_locator(plt.MultipleLocator(locators['yaxis.major']))
        self.ax.yaxis.set_minor_locator(plt.MultipleLocator(locators['yaxis.minor']))
        
    def set_axis_limits(self, xlim = None, ylim = None):
        self.xlim = xlim
        self.ylim = ylim
     
    def set_data_to_plot(self, data_to_plot):
        self.data_to_plot = data_to_plot

    def plot_lines(self):
        return [_ for line in self.data_to_plot 
                  for _ in plt.plot( line.data[:,0] / self.xaxis_factor, 
                                     line.data[:,1],
                                     color = line.color,
                                     lw = line.lw,
                                     alpha = line.alpha,
                                     linestyle = line.linestyle,
                                    )]
    
    def set_axis_labels(self, attr):
        plt.xlabel(u'$t$, $10^{0} \cdot$ atomic time unit'.format(
            int(np.log10(self.xaxis_factor))
        ))
        plt.ylabel(u'${0}$'.format(attr))
    
    def make_plot(self, xaxis_factor = 1):
        self.xaxis_factor = xaxis_factor
        
        self.plot_lines()
        
        print(self.xlim)
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)     
        