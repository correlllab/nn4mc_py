# Neural Networks for Microcontrollers: Python

[![Docs](https://readthedocs.org/projects/nn4mc/badge)](https://nn4mc.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/correlllab/nn4mc_py/blob/master/LICENSE.md)
[![ONXX Tutorial](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/correlllab/nn4mc_cpp/blob/master/examples/ONNX_NN4MC.ipynb)

This is a python implementation of Neural Networks for Microcontrollers (nn4mc) (originally implemented in C++) that allows for the translation of trained neural network models to C code for use in embedded systems.

## Development Status

Please note that nn4mc_py is still in development and may have many bugs. We are working hard on getting everything operating seamlessly, and feel free to reach out with any questions.

## Using nn4mc

### Installation

Simply use the Python package manager pip and run the following command.
```
pip install nn4mc
```

### Usage

You will most likely only need to import the translator module
```python
import nn4mc.translator as nnTr
```

Then you can translate a file with the following command
```python
nnTr.translate("path/to/file", 'hdf5', "output/path")
```

### What about packages other than Keras?

We currently develop instructionals on converting packages from multiple sources to Keras.

<img src="docs/assets/bitmap.png" align="center" width=100%/>

### Technical Questions

Please direct your technical questions to [Stack Overflow](https://stackoverflow.com) using the [nn4mc](https://stackoverflow.com/questions/tagged/nn4mc) tag or e-mail Sarah.AguasvivasManzano@colorado.edu. Also feel free to initiate a new issue in our github repository.

### Getting Involved

- For bug report or feature requests please submit a [GitHub issue](https://github.com/correlllab/nn4mc/issues).
- For contributions please submit a [pull request](https://github.com/correlllab/nn4mc/pulls)
### Citing nn4mc:

We encourage to use the following citation references for academic use of nn4mc.

**BibTeX citation:**

```
@misc{nn4mc,
        title={Embedded Neural Networks for Robot Autonomy},
        author={Sarah {Aguasvivas Manzano} and Dana Hughes and Cooper Simpson and Radhen Patel and Nikolaus Correll},
        year={2019},
	note={International Symposium of Robotics Research (ISRR 2019)},
        eprint={1911.03848},
        archivePrefix={arXiv},
        primaryClass={cs.RO}
    }
```

**APA Format:**

```
Manzano, S. A., Hughes, D., Simpson, C., Patel, R., & Correll, N. (2019). Embedded Neural Networks for Robot Autonomy. 
International Symposium of Robotics Research (ISRR 2019). arXiv preprint arXiv:1911.03848. 
```


### Contributors:

nn4mc is supported by the [Correll Lab](http://correll.cs.colorado.edu/) at the University of Colorado Boulder. We also receive support from the Airforce Office of Scientific Research (AFOSR), we are very grateful for this support. 