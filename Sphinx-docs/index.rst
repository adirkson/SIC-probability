.. Probabilistic SIC forecasts documentation master file, created by
   sphinx-quickstart on Mon Mar 19 11:00:46 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Calibrated Probabilistic Forecasts of Sea Ice Concentration
===========================================================
This module is for making probabilistic forecasts of sea ice concentration (SIC) using the methods described in Dirkson et al, 2018. The user should refer to this paper for further details.

It consists of two classes: :class:`beinf` and :class:`taqm`. The :class:`beinf` class is used to fit SIC data to the zero- and one- inflated beta (BEINF) distribution (Ospina and Ferrari, 2010), and to freeze BEINF distribution objects that can be used to compute e.g. its pdf, its cdf, random variates. The :class:`taqm` class is used to carry out the trend adjusted quantile mapping (TAQM) calibration method. Both of these classes are to be applied at the individual grid cell level, examples for which are included in the tutorial below. 

This project was built in Python v2.7 and relies on the classes `beinf.py` and `taqm.py` available at https://github.com/adirkson/SIC-probability.

Contents
---------

.. toctree::
   :maxdepth: 2

   tutorial
   code

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
-----------
Dirkson, A., Merryfield, W. J., & Monahan, A. (2018). Calibrated Probabilistic Forecasts of Sea Ice Concentration. `Journal of Climate`, submitted.

Ospina, R., & Ferrari, S. L. (2010). Inflated beta distributions. `Statistical Papers`, 51(1), 111.





