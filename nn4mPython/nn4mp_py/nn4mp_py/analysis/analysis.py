import numpy as np
import h5py

from keras import backend as K
from keras.models import load_model as k_load

class Analyzer:

    def __init__(self):
        self.old_model = None
        self.new_model = None

    def load_model(self, file_name):
        self.model = k_load(file_name)
        print(model.summary())

    def write_model(self, output_name):
        pass

    def cheapify(self, min_error):
        self.new_model = self.old_model

        for layer in self.new_model.layers:
            weights = layer.get_weights()
            #Preform svd and then..
            u,s,v = np.linalg.svd(weights[0])
            #Look at singular values in s and set some to zero.


            #Multiply u,s,v together to get original weight matrix
            weights[0] =

            layer.set_weights(weights)

    def compare(self):
        #Set up test here
        #NOTE: Will implement all layer output errors later
        '''
        input = self.old_model.input
        outputs = [layer.output for layer in self.old_model.layers]
        functors = [K.function([input], [output]) for output in outputs]
        '''
        input_shape = self.old_model.inputs
        test = np.ones((input_shape))

        #Then look at output of both
        old_output = self.old_model.predict(test)
        new_output = self.new_model.predict(test)

        print("Old output: ")
        print(old_output,'\n')

        print('New output: ')
        print(new_output,'\n')

        #Compute error
        np.linalg.norm(np.subtract(old_output-new_output))
