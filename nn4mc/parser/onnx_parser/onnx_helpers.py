import onnx
from onnx2keras import onnx_to_keras

# /*------------------------------------------------- onnx2keras -----
#          |  Function: onnx2keras
#          |
#          |  Purpose:  EXPLAIN WHAT THIS FUNCTION DOES TO SUPPORT THE CORRECT
#          |      OPERATION OF THE PROGRAM, AND HOW IT DOES IT.
#          |
#          |  Parameters:
#          |      parameter_name (IN, OUT, or IN/OUT) -- EXPLANATION OF THE
#          |              PURPOSE OF THIS PARAMETER TO THE FUNCTION.
#          |                      (REPEAT THIS FOR ALL FORMAL PARAMETERS OF
#          |                       THIS FUNCTION.
#          |                       IN = USED TO PASS DATA INTO THIS FUNCTION,
#          |                       OUT = USED TO PASS DATA OUT OF THIS FUNCTION
#          |                       IN/OUT = USED FOR BOTH PURPOSES.)
#          |
#          |  Returns:  IF THIS FUNCTION SENDS BACK A VALUE VIA THE RETURN
#          |      MECHANISM, DESCRIBE THE PURPOSE OF THAT VALUE HERE.
#          *-------------------------------------------------------------------*/


def onnx2keras(file):
    # Load ONNX model
    onnx_model = onnx.load(file)

    input_all = [node.name for node in onnx_model.graph.input]
    # print(str(input_all[0]))

    # Call the converter
    k_model = onnx_to_keras(onnx_model, [str(input_all[0])])

    return k_model