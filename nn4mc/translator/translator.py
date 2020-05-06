import nn4mc.parser as nnPr
import nn4mc.generator as nnGr

types = {'hdf5' : 'nnPr.HDF5Parser(',
        'pk' : 'nnPr.PYTorchParser('}

def translate(file_path, file_type, output_path, *, template='c_standard'):
    if file_type not in types:
        #Raise error
        pass

    parser = eval(types[file_type] + '"' + file_path + '")')
    parser.parse()

    generator = nnGr.Generator(parser.nn)
    generator.generate(output_path)

#This method is intended for nn4mc_web
#NOTE: Need to fix the eval thing given file object
def translatePlain(file_obj, file_type, *, template='c_standard'):
    if file_type not in types:
        #Raise error
        pass

    parser = nnPr.HDF5Parser(file_obj)
    parser.parse()

    generator = nnGr.Generator(parser.nn)
    output = generator.generate(output_type='D')

    return output
