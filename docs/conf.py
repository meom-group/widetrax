
import os
import sys

#sys.path.insert(0, os.path.abspath('..'))
#sys.path.extend([os.path.dirname(os.getcwd()), os.path.join(os.path.dirname(os.getcwd()), "widetrax")])
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "widetrax"))

autodoc_member_order = "groupwise"
exclude_patterns = ["_build", "**tests**"]

from version import __version__

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'widetrax'
copyright = '2024, Amine Ouhechou, Julien Le Sommer'
author = 'Amine Ouhechou, Julien Le Sommer'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "myst_parser"]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': True,
    'inherited-members': True,
    'show-inheritance': True
}
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
