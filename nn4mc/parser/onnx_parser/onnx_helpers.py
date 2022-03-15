import onnx
from onnx2keras import onnx_to_keras

def onnx2keras(file):
    # Load ONNX model
    onnx_model = onnx.load(file)

    input_all = [node.name for node in onnx_model.graph.input]
    # print(str(input_all[0]))

    # Call the converter
    k_model = onnx_to_keras(onnx_model, [str(input_all[0])])

    return k_model