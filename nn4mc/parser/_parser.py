from abc import ABC, abstractmethod
#Parent class for all parser objects
#NOTE: Seems a little useless, should maybe be changed.
class Parser(ABC):
    file_name = '' #Name of file to be scraped
    file_format = '' #File format to be scraped

    @abstractmethod
    def parse(self):
        pass
