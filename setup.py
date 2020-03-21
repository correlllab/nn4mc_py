from setuptools import setup, find_packages

with open('README.md', 'r') as ld:
    long_description = ld.read()

setup(name='nn4mc_py',
        version='0.2',
        description='Python package for generating microcontroller code from'
        'neural networks.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Cooper Simpson, Sarah Manzano, Dana Hughes',
        author_email='cooper.simpson@colorado.edu',
        url='https://github.com/RS-Coop/nn4mc_py',
        packages=find_packages(),
        install_requires=['h5py', 'numpy'],
        classifiers=[
                    'Programming Language :: Python :: 3',
                    'License :: OSI Approved :: MIT License',
                    'Operating System :: OS Independent'
        ]
        #scripts=['tests/neuralnetwork_test'],
)
