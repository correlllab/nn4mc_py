def processTemplate(self, contents):
    start = contents.find(self.START_DELIMITER)
    end = contents.find(self.END_DELIMITER)

    start += self.START_DELIMITER.len()

    contents = contents[start:end]

    contents.replace(self.WEIGHT_DATATYPE_DELIMITER, self.weight_datatype)
    contents.replace(self.INDEX_DATATYPE_DELIMITER, self.index_datatype)

    #Not sure about this one
    #contents.replace('<%LAYER_DATATYPE')

    return contents

def getFunctionStrings(self, contents):
    start = contents.find(self.INIT_START_DELIMITER)
    end = contents.find(self.INIT_END_DELIMITER)

    start += self.INIT_START_DELIMITER.len()

    init = contents[start:end]

    start = contents.find(self.FWD_START_DELIMITER)
    end = contents.find(self.FWD_END_DELIMITER)

    start += self.FWD_END_DELIMITER.len()

    fwd = contents[start:end]

    return init, fwd
    
def addLayer(self, node):
    layer_type = node.layer.identifier

    #Check if layer has already been processed.
    if layer_type in self.include_strings.keys():
        return
    #If not open associated layer header and process
    with open(self.template_include_dir + '/' + layer_type + '.h.template', 'r') as file:
        file_contents = file.readlines()

    #Replace delimiters
    file_contents = self.processTemplate(file_contents)

    self.include_files[layer_type] = file_contents

    #Opend associated layer source and process
    with open(self.template_source_dir + '/' + layer_type + '.c.template', 'r') as file:
        file_contents = file.readlines()

    #Extract init calls and fwd calls.
    init, fwd = self.getFunctionStrings(file_contents)

    self.init_strings[layer_type] = init
    self.fwd_strings[layer_type] = fwd

    #Replace delimiters
    file_contents = self.processTemplate(file_contents)

    self.source_files[layer_type] = file_contents

def dump(self, output_dir):
    pass
    #Writ out contents of include files and source files
