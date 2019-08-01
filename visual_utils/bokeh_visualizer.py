
import operator
import numpy as np
from collections import OrderedDict
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models.sources import ColumnDataSource
from bokeh.io import curdoc
from threading import Thread
import time
from tornado import gen
from functools import partial

class Visualizer():
    def __init__(self, best_ind, avg_ind):
        self.p = figure(title="evolution", x_axis_label="gen" ,y_axis_label ="energy",
           plot_width=400, plot_height=400)
        self.bar = figure(title="popul", x_axis_label="inds" ,y_axis_label ="energy",
           plot_width=400, plot_height=400) 
        best_energies = [ best_ind ]
        avg_energies = [ avg_ind ]
        index = [0]
        inds = [0 for x in range(0, 1)]
        index = [0] 
        self.inds_source = ColumnDataSource(data=dict(x=index,y=inds))
        self.bar.vbar(x="x", top="y",
                    source=self.inds_source, width=0.9)
        data = {'x' : index, 'y': best_energies, 'y2': avg_energies}
        self.source = ColumnDataSource(data)
        self.p.line(x='x', y='y', color="red", source = self.source)
        self.p.line(x='x', y='y2', color="green", source = self.source)
        self.doc = curdoc()
        self.layout = row(self.p, self.bar)
        #curdoc().add_periodic_callback(update, 50)
        self.doc.add_root(self.layout)

    @gen.coroutine
    def update(self,x, y, y2, inds):
        if (x == 1):
            self.source.data = dict(x=x, y=y, y2=y2)
        else:
            self.source.stream(dict(x=x, y=y, y2=y2))
        index = list(range(0, len(inds)))
        self.inds_source.data = dict(x=index,y=inds)
 
    def render_gen(self, gen, popul):
        scores = [x.score for x in popul]
        individuals = [x.genotype for x in popul]
        best_ind = np.min(scores)
        avg_ind = np.average(scores)
        new_list = [best_ind]
        new_avg_list = [avg_ind]
        data = {'x' : [gen], 'y':new_list, 'y2':new_avg_list}
        self.doc.add_next_tick_callback(partial(self.update, x = data['x'] ,
                                                y=data['y'],y2=data['y2'], inds=scores))
 
    def update_gen(self, gen, best_ind, avg_ind, popul):
        new_list = [best_ind]
        new_avg_list = [avg_ind]
        data = {'x' : [gen], 'y':new_list, 'y2':new_avg_list}
        self.doc.add_next_tick_callback(partial(self.update, x = data['x'] ,
                                                y=data['y'],y2=data['y2'], inds=popul))
               
    def test_run(self,source):
        init_b_values = source.data["y"]
        init_avg_values = source.data["y2"]
        while True:
            new_list = [init_b_values[-1] - 1]
            new_avg_list = [init_avg_values[-1] - 1]
            data = {'x' : [len(init_b_values) + 1], 'y':new_list, 'y2':new_avg_list}
            self.doc.add_next_tick_callback(partial(self.update, x = data['x'] , y=data['y'],y2=data['y2']))
            init_b_values = init_b_values + new_list
            init_avg_values = init_avg_values + new_avg_list
            time.sleep(5)


class Algorithm():
    def __init__(self, visualizer, popul):
        self.visualizer = visualizer
        self.popul = popul
        
    def run(self):
        max_gens = 1000
        best_ind = np.min(self.popul)
        avg_ind = np.average(self.popul)
        for gen in range(1, 1000):
            for i, p in enumerate(self.popul):
                self.popul[i] = p - 1
                
            best_ind = np.min(self.popul)
            avg_ind = np.average(self.popul)
            self.visualizer.update_gen(gen, best_ind, avg_ind, self.popul)
            time.sleep(2)

# def main():
#     popul = [np.random.randint(-10,10) for x in range(0,10)]
#     best_ind = np.max(popul)
#     avg_ind = np.average(popul)
#     visualizer = Visualizer(best_ind, avg_ind)
#     alg = Algorithm(visualizer, popul)
#     thread = Thread(target=alg.run)
#     thread.start()

# if __name__=="__main__":
#     main()

