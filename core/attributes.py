# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013-2014)
# All rights reserved
# created:          Dec 23rd, 2013
# last-modified:    Apr 18th, 2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from ..utility.log_report import Log
from ..utility.common import Callable

## attribute initialization function (with default value when not initialized) 
#  @param attrs is an attribute dictionnary
#  @param attr_name is the name of the attribute to initialize
#  @param default_value [None] is the value to be associated with the attribute if no value is found within attrs
#  @param required [False] indicate whether the attribute may be omitted or not 
def attr_init(attrs, attr_name, default_value = None, required = False):
    if attr_name in attrs:
        return attrs[attr_name]
    else:
        if not "__copy" in attrs and required:
            Log.report(Log.Error, "required argument %s is absent" % attr_name) 
        else:
            return default_value


## Debug attributes class to adapt the debug display message properties 
#  @param display_format C string used when displaying debug message
#  @param color of the debug message
#  @param pre_process  pre_process function to be applied to the Node before display
#  @param require_header list of headers required to generate the debug message
class ML_Debug:
    
    ## initialization of a new ML_Debug object
    def __init__(self, display_format = None, color = None, pre_process = lambda v: v, require_header = []):
        self.display_format = display_format
        self.color = color
        self.pre_process = pre_process
        self.require_header = require_header

    def get_display_format(self, default = "%f"):
        return self.display_format if self.display_format else default

    def get_pre_process(self, value_to_display, optree):
        return self.pre_process(value_to_display)

    def get_require_header(self):
        return self.require_header

    def select_object(self, optree):
        return self

## debug object which automatically adapt to input parameters 
class ML_MultiDebug(ML_Debug):
  def __init__(self, debug_object_map, key_function = lambda optree: optree.get_precision()):
    self.debug_object_map = debug_object_map
    self.key_function = key_function

  def select_object(self, optree): 
    return self.debug_object_map[self.key_function(optree)]

class ML_AdvancedDebug(ML_Debug):
  def get_pre_process(self, value_to_display, optree):
    return self.pre_process(value_to_display, optree)


## Object to keep track of ML's node accross the several optimizations passes
class Handle: 
    def __init__(self, node = None):
        self.node = node

    def set_node(self, new_node):
        self.node = new_node

    def get_node(self):
        return self.node


## Base class to store Node's attributes
class Attributes: 
    """ Attribute management class for Metalibm's Operation """
    default_precision     = [None]
    default_rounding_mode = [None]
    default_silent        = [None]
    str_del               = "| "

    def __init__(self, **init_map):
        self.precision  = attr_init(init_map, "precision", Attributes.default_precision[0])
        self.interval   = attr_init(init_map, "interval")
        self.debug      = attr_init(init_map, "debug")
        self.exact      = attr_init(init_map, "exact")
        self.tag        = attr_init(init_map, "tag")
        self.max_abs_error = attr_init(init_map, "max_abs_error")
        self.silent     = attr_init(init_map, "silent", Attributes.default_silent[0])
        self.handle     = attr_init(init_map, "handle", Handle())
        self.clearprevious = attr_init(init_map, "clearprevious")
        # rounding mode (if applicable) of the operation
        self.rounding_mode = attr_init(init_map, "rounding_mode", Attributes.default_rounding_mode[0])
        self.rounding_mode_dependant = None
        self.prevent_optimization = attr_init(init_map, "prevent_optimization")
        self.unbreakable  = attr_init(init_map, "unbreakable", False)


    def get_str(self, tab_level = 0):
        """ string conversion for operation graph 
            depth:                  number of level to be crossed (None: infty)
            display_precision:      enable/display format display
        """
        tab_str   = Attributes.str_del * tab_level
        debug_str = "T" if self.debug else "F"
        return tab_str + "%s D[%s] E[%s] RND=%s[%s]" % (self.interval, debug_str, self.exact, self.rounding_mode, self.rounding_mode_dependant)


    def get_copy(self):
        return Attributes(precision = self.precision, interval = self.interval, debug = self.debug, exact = self.exact, tag = self.tag, max_abs_error = self.max_abs_error, silent = self.silent, handle = self.handle, clearprevious = self.clearprevious, rounding_mode = self.rounding_mode, prevent_optimization = self.prevent_optimization)

    def get_light_copy(self):
        return Attributes(precision = self.precision, debug = self.debug, tag = self.tag, silent = self.silent, handle = self.handle, clearprevious = self.clearprevious, rounding_mode = self.rounding_mode, prevent_optimization = self.prevent_optimization)

    def set_attr(self, **init_map):
        """ generic attribute setter """
        for attr_name in init_map:
            attr_value = init_map[attr_name]
            setattr(self, attr_name, attr_value)

    def get_prevent_optimization(self):
        return self.prevent_optimization
    def set_prevent_optimization(self, prevent_optimization):
        self.prevent_optimization = prevent_optimization

    def get_unbreakable(self):
        """ unbreakable getter """
        return self.unbreakable
    def set_unbreakable(self, new_breakable):
        """ unbreakable setter """
        self.unbreakable = new_breakable

    def set_silent(self, new_silent):
        self.silent = new_silent
    def get_silent(self):
        return self.silent

    def set_precision(self, new_precision):
        """ precision setter """
        self.precision = new_precision
    def get_precision(self):
        """ precision getter """
        return self.precision

    def set_exact(self, new_exact):
        """ exact setter """
        self.exact = new_exact
    def get_exact(self):
        """ exact getter """
        return self.exact


    def set_interval(self, interval):
        """ interval setter """
        self.interval = interval
    def get_interval(self):
        """ interval getter """
        return self.interval 


    def set_tag(self, new_tag):
        """ tag setter """
        self.tag = new_tag
    def get_tag(self):
        """ tag getter """
        return self.tag


    def set_debug(self, new_debug):
        """ debug setter """
        self.debug = new_debug
    def get_debug(self):
        """ debug getter """
        return self.debug

    def set_handle(self, new_handle):
        """ handle setter """
        self.handle = new_handle
    def get_handle(self):
        """ handle getter """
        return self.handle

    def get_clearprevious(self):
        return self.clearprevious
    def set_clearprevious(self, new_clearprevious):
        self.clearprevious = new_clearprevious

    def get_rounding_mode(self):
        return self.rounding_mode
    def set_rounding_mode(self, new_rounding_mode):
        self.rounding_mode = new_rounding_mode


    def get_max_abs_error(self):
        return self.max_abs_error
    def set_max_abs_error(self, new_max_abs_error):
        self.max_abs_error = new_max_abs_error


    # static method definition
    def set_default_precision(new_precision):
        Attributes.default_precision.insert(0, new_precision)
    def unset_default_precision():
        Attributes.default_precision.pop(0)
        if len(Attributes.default_precision) < 1: raise Exception()

    def set_default_rounding_mode(new_rounding_mode):
        Attributes.default_rounding_mode.insert(0, new_rounding_mode)
    def unset_default_rounding_mode():  
        Attributes.default_rounding_mode.pop(0)
        if len(Attributes.default_rounding_mode) < 1: raise Exception()

    def set_default_silent(new_silent_value):
        Attributes.default_silent.insert(0, new_silent_value)
    def unset_default_silent():
        Attributes.default_silent.pop(0)
        if len(Attributes.default_silent) < 1: raise Exception()

    # static method bindings
    set_default_precision = Callable(set_default_precision)
    unset_default_precision = Callable(unset_default_precision)

    set_default_rounding_mode = Callable(set_default_rounding_mode)
    unset_default_rounding_mode = Callable(unset_default_rounding_mode)

    set_default_silent = Callable(set_default_silent)
    unset_default_silent = Callable(unset_default_silent)

