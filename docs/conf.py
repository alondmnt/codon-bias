# Configuration file for the Sphinx documentation builder.
import os
import sys

# -- Project information -----------------------------------------------------
project = 'codon-bias'
copyright = '2024, Alon Diament'
author = 'Alon Diament'
release = '0.3.4'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Automatically include docstrings
    'sphinx.ext.napoleon',     # Support for Google/NumPy-style docstrings
    'sphinx.ext.viewcode',     # Add links to source code
    'sphinx.ext.githubpages',  # Create .nojekyll file for GitHub Pages
    'nbsphinx',               # Support for Jupyter notebooks
]

# -- Options for nbsphinx ---------------------------------------------------
nbsphinx_allow_errors = True   # Continue building even if there are execution errors

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'  # Read the Docs theme
html_static_path = ['_static']

# If your documentation needs to find your Python modules
sys.path.insert(0, os.path.abspath('..'))
