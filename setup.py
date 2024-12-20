from setuptools import setup

setup(
    name='codon-bias',
    description='codon usage bias analysis tools',
    long_description_content_type = 'text/markdown',
    url='https://github.com/alondmnt/codon-bias',
    project_urls={'Documentation': 'https://codon-bias.readthedocs.io/en/latest/'},
    author='Alon Diament',
    author_email='dev@alondmnt.com',
    license='MIT',
    packages=['codonbias'],
    package_data={'': ['*.csv']},
    include_package_data=True,
    python_requires='>=3',
    install_requires=['numpy',
                      'scipy',
                      'pandas',
                      'pandarallel'
                      ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
)
