from setuptools import setup, find_packages

with open('README.md', 'r') as ld:
    long_description = ld.read()

setup(name='nn4mc',
        version='0.2.7',
        description='Neural Networks for Microcontrollers (nn4mc) is a Python package '
        'for generating microcontroller code in c from pre-trained models. Our intended '
        'audience is roboticists looking to embed intelligence into their applications; '
        'however, anyone interested in using a neural network in a microcontroller will '
        'find this software useful.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Cooper Simpson, Sarah Manzano, Dana Hughes',
        author_email='cooper.simpson@colorado.edu',
        url='https://github.com/correlllab/nn4mc_py',
        license="MIT",
        packages=find_packages(exclude=('tests','tests.*')),
        include_package_data=True,
        install_requires=['h5py', 'numpy'], #Might need more here
        classifiers=[
                    'Programming Language :: Python :: 3',
                    'License :: OSI Approved :: MIT License',
                    'Operating System :: OS Independent',
                    'Development Status :: 3 - Alpha',
                    'Topic :: Software Development :: Code Generators'
                    ],
        keywords=['Neural Network', 'Microcontroller']
)
