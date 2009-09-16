.. Grammar-guided Feature Extraction documentation master file, created by
   sphinx-quickstart on Sat Sep 12 08:57:59 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GGFE User Guide and API Documentation
=====================================

This resource describes how to use the Grammar-guided Feature
Extraction (GGFE) software. GGFE is a tool for randomly generating
features from a stochastic generative grammar, also known as a feature grammar.
Feature grammars can be expressed as a pure Python idiom,

.. sourcecode:: python

   # Define the grammar.
   Feature[X] = erode[X] |
                dilate[X]

This grammar has a single production ``Feature`` with two rules, ``erode``
and ``dilate``. Non-terminal productions can take any number of arguments,
as the ``Feature``. Although not required, *non-terminal* productions are 
*capitalized* and terminal functions (such as ``erode`` and ``dilate``) are in
lowercase.

Please see the :ref:`tutorial` to learn more about feature grammars.

Contents:

.. toctree::
   :maxdepth: 2

   tutorial.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

