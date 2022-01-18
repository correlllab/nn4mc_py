from queue import Queue
from nn4mc.datastructures import Layer

#Auxillary data structure for nodes in graph representation.
class LayerNode:
    def __init__(self, layer):
        self.layer = layer
        self.visited = False

    #Hashes on the identifier string.
    def __hash__(self):
        return hash(self.layer.identifier)

    #Equality can be to another LayerNode or just a Layer.
    def __eq__(self,other):
        if isinstance(other, Layer):
            return self.layer.identifier == other.identifier

        elif isinstance(other,LayerNode):
            return self.layer.identifier == other.layer.identifier

################################################################################
#Graph datastructure to represent neural network connection structure.
#Creates common representation to be iterated on by code generator downstream.
#NOTE:
class NeuralNetwork:
    def __init__(self):
        self.layer_list = []
        self.layers = {} #Dictionary of LayerNodes and list of edges
        self.input = [] #List of input LayerNodes

    #Returns a layer if it exists
    def getLayer(self, id):
        for layernode in self.layers:
            if layernode.layer.identifier == id:
                return layernode.layer
        return None

    def setLayer(self, id: str, new_layer: Layer):
        for layernode in self.layers:
            if layernode.layer.identifier == id:
                layernode.layer = new_layer

    #Adds LayerNode to graph with empty list of edges
    def addLayer(self, layer):
        newLayer = LayerNode(layer) #New layer
        self.layer_list += [layer]
        self.layers[newLayer] = [] #No edges

        if(layer.isInput()): #Adds to input list if Input Layer
            self.input.append(newLayer)

    #Adds new edge between two layers
    #NOTE: Start and end are Layer objects
    #NOTE: Not dealing with undefined edges (i.e start or end not existing)
    def addEdge(self, start, end):
        for layernode in self.layers: #Find the LayerNode associated with end
            if layernode == end:
                node = layernode

        self.layers[start].append(node) #Add LayerNode to starts list

    #Iterator for the graph datastructure
    #Uses BFS to search the graph and yields nodes as they are found
    #NOTE: Not dealing with anything but sequential model (graph is overkill)
    #NOTE: Could be edited to deal with non-sequential model
    def iterate(self):
        for node in self.layers:
            node.visited = False
        q = Queue()
        for node in self.input: #Add all input nodes
            node.visited = True
            q.put(node)
        while not q.empty():
            node = q.get()

            for edge in self.layers[node]:
                if edge.visited == False:
                    edge.visited = True
                    q.put(edge)
            yield node #Returns a node

    def iterate_layer_list(self):
        for layer in self.layer_list:
            yield layer
