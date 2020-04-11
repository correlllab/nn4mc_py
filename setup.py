from setuptools import setup, find_packages

with open('README.md', 'r') as ld:
    long_description = ld.read()

setup(name='nn4mc',
        version='0.1.0',
        description='Neural Networks for Microcontrollers is a Python package'
        'for generating microcontroller code in c from traind models.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Cooper Simpson, Sarah Manzano, Dana Hughes',
        author_email='cooper.simpson@colorado.edu',
        url='https://github.com/RS-Coop/nn4mc_py',
        # download_url = '',
        liscense="MIT",
        # packages=[find_packages(exclude=('tests',))],
        packages=['nn4mc_py'],
        include_package_data=True,
        install_requires=['h5py', 'numpy'], #Might need more here
        classifiers=[
                    'Programming Language :: Python :: 3',
                    'License :: OSI Approved :: MIT License',
                    'Operating System :: OS Independent',
                    'Development Status :: 3 - Alpha',
                    'Topic :: Neural Networks :: Robotics'
                    ],
        keywords=['Neural Network', 'Microcontroller']
)
