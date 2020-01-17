from ._parser import Parser

class JSONParser(Parser):
    def __init__(self, file_name):
        self.file_name = file_name
