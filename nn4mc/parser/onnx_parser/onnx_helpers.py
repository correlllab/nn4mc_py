import onnx
from onnx2keras import onnx_to_keras

# /*------------------------------------------------- onnx2keras -----
#          |  Function: onnx2keras
#          |
#          |  Purpose:  CONVERTING AN ONNX MODEL TO KERA VERSION
#          |
#          |  Parameters: .onnx file
#          |
#          |  Returns: Kera model
#          *-------------------------------------------------------------------*/


def onnx2keras(file):
    # Load ONNX model
    onnx_model = onnx.load(file)

    input_all = [node.name for node in onnx_model.graph.input]

    # Call the converter
    k_model = onnx_to_keras(onnx_model, [input_all[0]])


    return k_model