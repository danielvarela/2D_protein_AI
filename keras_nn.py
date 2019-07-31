from keras.models import Sequential
from keras.layers import Dense
from ann_visualizer.visualize import ann_viz
import numpy as np

class FFmodel():
    def __init__(self, input_size = 4):
        self.model = Sequential()
        self.input_size= input_size
        self.hidden_size = 100
        self.output_size = 4
        self.model.add(Dense(self.hidden_size, input_dim=self.input_size, activation='sigmoid'))
        self.model.add(Dense(self.output_size, input_dim=self.hidden_size, activation='sigmoid'))
        self.model.compile(loss='mean_squared_error', optimizer='sgd')
        w = self.model.get_weights()
        self.sizes = []
        self.sizes = [[self.input_size,self.hidden_size],[1,self.hidden_size],[self.hidden_size,self.output_size],[1,self.output_size]]
        self.size = (self.input_size*self.hidden_size+self.hidden_size+self.hidden_size*self.output_size+self.output_size)

    def render(self):
        print(self.model.summary())
        ann_viz(self.model, title="ANN visualization")

    def nn_connections(self):
        return self.size
    
    def get_weights(self):
        w = self.get_weights()
        weights = w.ravel()
        return weights

    def set_weights(self, values):
        w = []
        w_d = self.model.get_weights()
        for d,v in self.sizes:
            w_list = []
            for i in range(0, d):
                l = np.array(values[:v], dtype="float32")
                w_list.append(l)
                values = values[v:]
            if (d > 2):
                w.append(np.array(w_list, dtype="float32"))
            else:
                # bias weights
                w.append(l)
            
        self.model.set_weights(w)

    def run(self,inputs):
        #i = np.reshape(inputs,(4,1) )
        #i = [np.array(x, dtype="float32") for x in i]
        i = np.reshape(inputs, (1,self.input_size))
        return self.model.predict(x=i)[0]

    def predict(self, i):
        return self.model.predict(x=i)
