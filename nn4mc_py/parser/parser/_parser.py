class Parser:
    def __init__(self):
        self.file_name = ''
        self.file_format = ''

class Opdata:
    pass
    #Maybe declare vaiables here or maybe not.

class OpdataWeights:
    pass
    #Maybe declare variables here or maybe not.

class HDF5Parser(Parser):
    def __init__(self, file_name):
        self.file_name = file_name

        self.layer_builder_list = []
        self.layer_IDs = []
        self.layer_edges = []

        self.builder_map = {}
        self.layer_map = {}


    #Imported Functions
    from ._HDF5Parser import *

class JSONParser(Parser):
    def __init__(self, file_name):
        self.file_name = file_name

    #Imported Functions
    from ._JSONParser import .