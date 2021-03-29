# -*- coding: utf-8 -*-
"""DSTTools

This module provides a set of tools in Python to work with register data at Denmark Statistics.

A **dataset** is defined by a dictionary containing:

vars: list of variable names in register and after fetch, e.g. [('name of variable in register','name of variable after fetch'),...].
years: list of first and last year, [first_year, last_year], or string 'all'.
overwrite: boolean whether to overwrite data which has already been fetched.

For the register 'bef' this could be:

bef_vars = [('pnr',),('befSourceYear','year'),('koen','sex'),('foed_dag','birthday')]
bef = {'name':'bef','vars':bef_vars,'years':'all',overwrite:False}
bef_selected_years = {'name':'bef','vars':bef_vars,'years':[1986,2015],overwrite:False}

Note: If something goes wrong while running stuff in parallel, terminate the Jupyter session.

"""

from .info import *
from .fetch_from_sas import *
from .convert_from_sas import *
from .combine_years import *
from .analysis import *