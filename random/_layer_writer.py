class LayerWriter():
    init_template = ''

    def write_init(self):
        pass

    def build_map(self):
        pass

class Conv1DWriter(LayerWriter):
    pass

class Conv2DWriter(LayerWriter):
    pass

class DenseWriter(LayerWriter):
    pass

class FlattenWriter(LayerWriter):
    pass

class MaxPooling1DWriter(LayerWriter):
    pass

class MaxPooling2DWriter(LayerWriter):
    pass

class DropoutWriter(LayerWriter):
    pass

class SimpleRNNWriter(LayerWriter):
    pass

class GRUWriter(LayerWriter):
    pass

class LSTMWriter(LayerWriter):
    pass

class InputWriter(LayerWriter):
    pass

class ActivationWriter(LayerWriter):
    pass
