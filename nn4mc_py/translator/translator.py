import nn4mc_py.parser as nnPr
import nn4mc_py.generator as nnGr

types = {'hdf5' : 'nnPr.HDF5Parser(',
        'pk' : 'nnPr.PYTorchParser('}

def translate(file_path='', file_type='', output_path='', template='c_standard'):
    if file_type not in types:
        #Raise error
        pass

    parser = eval(types[file_type] + '"' + file_path + '")')
    parser.parse()

    generator = nnGr.Generator(parser.nn, output_path)
    generator.generate()
