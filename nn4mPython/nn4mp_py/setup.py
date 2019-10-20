from setuptools import setup, find_packages

setup(name='nn4mp',
        version='0.1',
        description='Python package for generating microcontroller code from'
        'neural networks.',
        #long_description=read('README'),
        author='Cooper Simpson, Sarah Manzano, Dana Hughes',
        #package_dir = {'':'nn4mp'},
        packages=find_packages(),
        install_requires=['h5py'],
        scripts=['tests/neuralnetwork_test']
        )
