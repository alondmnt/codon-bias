from setuptools import setup

setup(
    name='codon-bias',
    description='codon usage bias analysis tools',
    url='https://github.com/alondmnt/codon-bias',
    author='Alon Diament',
    author_email='dev@alondmnt.com',
    license='MIT',
    packages=['codonbias'],
    package_data={'': ['*.csv']},
    include_package_data=True,
    install_requires=['numpy',
                      'scipy',
                      'pandas',
                      ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
