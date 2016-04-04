# -*- coding: utf-8 -*-

import sollya

def ulp(v, format_):
    """ return a 'unit in last place' value for <v> assuming precision is defined by format _ """
    return sollya.S2**(sollya.ceil(sollya.log2(sollya.abs(v))) - (format_.get_precision() + 1))

