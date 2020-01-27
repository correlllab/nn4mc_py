from queue import Queue
from nn4mc_py.datastructures import layer

class LayerNode: #Class to hold Layer and other data.
    def __init__(self, layer):
        self.visited = False
        self.layer = layer

    def __hash__(self): #Hashes on the identifier string
        return hash(self.layer.identifier)

    def __eq__(self,other): #Can be equal to another LayerNode or just a Layer
        if isinstance(other, layer.Layer):
            return self.layer.identifier == other.identifier

        elif isinstance(other,LayerNode):
            return self.layer.identifier == other.layer.identifier

class NeuralNetwork: #Graph data structure
    def __init__(self):
        self.layers = {} #Dictionary of LayerNodes and list of edges
        self.input = [] #List of input LayerNodes

    def getLayer(self, id):
        for layernode in self.layers.keys():
            if layernode.layer.identifier == id:
                return layernode.layer

        return None

    def addLayer(self, layer): #Adds LayerNode to dict with empty list as value
        newLayer = LayerNode(layer)
        self.layers[newLayer] = []

        if(layer.isInput()): #Adds to input list if it is Input Layer
            self.input.append(newLayer)

    #NOTE: Start and end are Layer objects
    def addEdge(self, start, end): #Adds edge between two LayerNodes with corresponding Layers
        for key in self.layers: #Find the LayerNode associated with end
            if key == end:
                node = key

        self.layers[start].append(node) #Add LayerNode to starts list

    #NOTE: Start and end are id's
    def addEdgeID(self, start, end):
        pass


    def iterate(self): #Essentially a BFS which uses yield on each loop.
        for node in self.layers:
            node.visited = False

        q = Queue()

        for node in self.input: #Add all input nodes
            node.visited = True
            q.put(node)

        while q.empty() == False:
            node = q.get()

            for edge in self.layers[node]:
                if edge.visited == False:
                    edge.visited = True
                    q.put(edge)

            yield node
