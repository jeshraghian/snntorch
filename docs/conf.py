#!/usr/bin/env python
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import snntorch

# fmt: off
__version__ = '0.4.0'
# fmt: on


# -- General configuration ---------------------------------------------

# needs_sphinx = '1.0'


# extensions = ["sphinx.ext.autodoc",
#               "sphinx.ext.viewcode",
#               "sphinx.ext.intersphinx",
#               "sphinx.ext.autodoc",
#               "sphinx.ext.mathjax",
#               "sphinx.ext.viewcode"]

extensions = ["sphinx.ext.autodoc", "sphinx.ext.viewcode"]

templates_path = ["_templates"]

# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "snntorch"
copyright = "2021, Jason K. Eshraghian"
author = "Jason K. Eshraghian"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = __version__
# version = "0.1.0"
# The full version, including alpha/beta/rc tags.
release = __version__
# release = "0.1.0"

language = None

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output -------------------------------------------

# html_theme = "alabaster"
html_theme = "sphinx_rtd_theme"
# html_theme = "sphinx_typo3_theme"


# html_theme_options = {}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/img/snntorch_alpha_full.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ["_static"]
# html_style = "css/default.css"

# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "snntorchdoc"


# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "snntorch.tex",
        "snntorch Documentation",
        "Jason K. Eshraghian",
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "snntorch", "snntorch Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "snntorch",
        "snntorch Documentation",
        author,
        "snntorch",
        "One line description of project.",
        "Miscellaneous",
    ),
]
