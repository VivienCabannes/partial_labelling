Partial Labelling Approached with through Structured Prediction (PLASP)
=======================================================================
:Topic: Generic implementations of weakly supervision algorithms
   developped in [CAB20]_, [CAB21a]_, [CAB21b]_.
:Author: Vivien Cabannes
:Version: 1.0.0 of 2021/06/07

Installation
------------
From wheel
~~~~~~~~~~
You can download our package from its pypi repository.

.. code:: shell

   $ pip install plasp

From source
~~~~~~~~~~~
You can download source code at https://github.com/VivienCabannes/partial_labelling/archive/master.zip.
Once download, our packages can be install through the following command.

.. code:: shell

   $ python <path to code folder>/setup.py install

You can also install it in develop mode, eventually with pip

.. code:: shell

    $ cd <path to code folder>
    $ pip install -e .

Usage
-----
See files:
 - ``problems/classification/libsvm_experiments.py``
 - ``problems/classification/semi_supervision_experiments.py``
 - and more generally ``*_experiements.py``

Package Requirements
--------------------
Most of the code is based on the following python libraries:
 - numpy
 - numba
 - matplotlib
 
Some testing done with notebook are based on:
 - jupyter-notebook
 - ipywidgets

For ranking, we used the following lp solver library:
 - cplex

To load LIBSVM files, more precisely to read libsvm files format we used:
 - scikit-learn
 
To load MULAN files, more precisely to read mulan files format we used:
 - arff
 - skmultilearn

Datasets links
--------------
Datasets can be download at:
 - LIBSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/oldfiles/index-1.0.html
 - MULAN: http://mulan.sourceforge.net/datasets-mlc.html

Change path in config file ``dataloader/config.py`` to specify path to your data.

References
----------
.. [CAB20] Structured Prediction with Partial Labelling through the Infimum Loss,
   Cabannes et al., *ICML*, 2020

.. [CAB21a] Disambiguation of weak supervision with exponential convergence rates,
   Cabannes et al., *ICML*, 2021

.. [CAB21b] Overcoming the curse of dimensionality with Laplacian regularization
   in semi-supervised learning, Cabannes et al., *NeurIPS*, 2021
