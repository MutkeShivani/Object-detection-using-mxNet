import mxnet as mx
import numpy as np
import cv2
from collections import namedtuple

def loadInceptionv3():
        #load the model from its saved state
        #MXNet calls this a checkpoint
        # In return, we get the input Symbol and the model parameters
        sym, arg_params, aux_params = mx.model.load_checkpoint('Inception-BN', 0)
        mod = mx.mod.Module(symbol=sym) # create a new Module and assign it the input Symbol.
        mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))]) #bind the input Symbol to input data
        mod.set_params(arg_params, aux_params) #set the model parameters.
        return mod

def loadCategories():
        synsetfile = open('synset.txt', 'r')
        synsets = []
        for l in synsetfile:
                synsets.append(l.rstrip())
        return synsets

def prepareNDArray(filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224,))
        img = np.swapaxes(img, 0, 2) #reshape the array from (image height, image width, 3) to (3, image height, image width)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :] #add a fourth dimension and build the NDArray
        return mx.nd.array(img)

def predict(filename, model, categories, n):
        array = prepareNDArray(filename)
        Batch = namedtuple('Batch', ['data'])
        model.forward(Batch([array]))
        prob = model.get_outputs()[0].asnumpy() #The model will output an NDArray holding the 1000 probabilities, corresponding to the 1000 categories. It has only one line since batch size is equal to 1
        prob = np.squeeze(prob) #an array with squeeze()
        sortedprobindex = np.argsort(prob)[::-1] #creating a second array holding the index of these probabilities sorted in descending order
        topn = []
        for i in sortedprobindex[0:n]:
                topn.append((prob[i], categories[i]))
        return topn

def init():
        model = loadInceptionv3()
        cats = loadCategories()
        return model, cats

m,c = init()
classs = predict("original.jpg",m,c,10)
print(classs)