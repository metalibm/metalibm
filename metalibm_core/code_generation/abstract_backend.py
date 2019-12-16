# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2018 Kalray
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################
# created:          Nov 17th, 2016
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nbrunie@kalray.eu)
###############################################################################

import inspect


from ..utility.log_report import Log
from .generator_utility import MatchResult, is_impl_list, ImplemList
from .code_constant import C_Code

LOG_BACKEND_INIT = Log.LogLevel(Log.Info, "backend_init")

## abstract backend class
class AbstractBackend(object):
    """ base abstract processor """
    target_name = "abstract"

    # does the platform support binary embedding in python module
    support_embedded_bin = False
    # is the platform native (same platform as the one used to execute
    # metalibm or is it a remote platform)
    cross_platform = True

    def __init__(self, *args):
        # create ordered list of parent architecture instances
        parent_class_list = get_parent_proc_class_list(self.__class__)
        self.parent_architecture = [parent(*args) for parent in create_proc_hierarchy(parent_class_list, [])]

        # create simplified of operation supported by the processor hierarchy
        self.simplified_rec_op_map = {}
        self.simplified_rec_op_map[C_Code] = self.generate_supported_op_map(language = C_Code)

    def __str__(self):
        """ Nice description string """
        return self.target_name

    def __repr__(self):
        return self.target_name

    ## return the backend target name
    def get_target_name(sef):
        return self.target_name

    def generate_expr(self, code_generator, code_object, optree, arg_tuple, **kwords):
        """ processor generate expression """
        language = kwords["language"] if "language" in kwords else C_Code
        implementation = self.get_recursive_implementation(optree, language)
        return implementation.generate_expr(code_generator, code_object, optree, arg_tuple, **kwords)

    def generate_supported_op_map(self, language = C_Code, table_getter = lambda self: self.code_generation_table):
        """ generate a map of every operations supported by the processor hierarchy,
            to be used in OptimizationEngine step """
        op_map = {}
        for parent_proc in self.parent_architecture:
            parent_proc.generate_local_op_map(language, op_map, table_getter = table_getter)
        # add locally supported operation last to patch
        # any previously registered support mapping
        self.generate_local_op_map(language, op_map)
        return op_map

    def generate_local_op_map(self, language = C_Code, op_map = None, table_getter = lambda self: self.code_generation_table):
        """ generate simplified map of locally supported operations """
        op_map = {} if op_map is None else op_map
        table = table_getter(self)
        if not language in table:
          return op_map
        else:
          local_map = table[language]
          for operation in local_map:
              if not operation in op_map: 
                  op_map[operation] = {}
              for specifier in local_map[operation]:
                  if not specifier in op_map[operation]: 
                      op_map[operation][specifier] = {}
                  for condition in local_map[operation][specifier]:
                      if not condition in op_map[operation][specifier]:
                          op_map[operation][specifier][condition] = {}
                      for interface_format in local_map[operation][specifier][condition]:
                          op_map[operation][specifier][condition][interface_format] = ML_FullySupported
          return op_map

    def get_implementation(self, optree, language = C_Code, table_getter = lambda self: self.code_generation_table, key_getter = lambda self, optree: self.get_operation_keys(optree)):
        """ return <self> implementation of operation performed by <optree> """
        #key_getter = AbstractBackend.get_operation_keys if key_getter is None else key_getter
        table = table_getter(self)
        op_class, interface, codegen_key = key_getter(self, optree)
        possible_impl = ImplemList()
        for condition in table[language][op_class][codegen_key]:
            if condition(optree):
                for interface_condition in table[language][op_class][codegen_key][condition]:
                    possible_match = interface_condition(*interface, optree=optree)
                    if possible_match:
                        implementation = table[language][op_class][codegen_key][condition][interface_condition]
                        sourceinfo = implementation.get_source_info()
                        Log.report(
                            Log.Verbose,
                            "optree {} matched to implementation @ {}",
                            optree,
                            str(sourceinfo)
                        )
                        if possible_match is True:
                            return implementation
                        elif isinstance(possible_match, MatchResult):
                            if possible_match.weak:
                                Log.report(Log.Verbose, "found weak match")
                                possible_impl.append((possible_match, implementation))
                            else:
                                return implementation
                        else:
                            Log.report(Log.Error, "unsupported match result: {} for optree {}", possible_match, optree)
        return possible_impl

    def get_recursive_implementation(self, optree, language=None,
                                     table_getter=lambda self: self.code_generation_table,
                                     key_getter=lambda self, optree: self.get_operation_keys(optree)):
        """ recursively search for an implementation of optree in the processor
            class hierarchy """
        impl_list = ImplemList()
        if self.is_local_supported_operation(optree, language = language, table_getter = table_getter, key_getter = key_getter):
            local_implementation = self.get_implementation(optree, language, table_getter = table_getter, key_getter = key_getter)
            # check if local_implementation is a list of weak implementation
            # or a single "string implementation
            if is_impl_list(local_implementation):
                impl_list += local_implementation
            else:
                return local_implementation
        for parent_proc in self.parent_architecture:
            if parent_proc.is_local_supported_operation(optree, language = language, table_getter = table_getter, key_getter = key_getter):
                parent_implementation = parent_proc.get_implementation(optree, language, table_getter = table_getter, key_getter = key_getter)
                if is_impl_list(parent_implementation):
                    impl_list += parent_implementation
                else:
                    return parent_implementation
        if len(impl_list) > 0:
            # select the first weak implementation match
            match, implementation = impl_list[0]
            return implementation

        # no implementation were found
        Log.report(Log.Verbose, "Tested architecture(s) for language %s:" % language)
        for parent_proc in self.parent_architecture:
          Log.report(Log.Verbose, "  %s " % parent_proc)
        Log.report(
            Log.Error,
            "the following operation is not supported by {}: \n{}",
            self.__class__,
            optree
        )

    def is_map_supported_operation(self, op_map, optree, language = C_Code, debug = False,  key_getter = lambda self, optree: self.get_operation_keys(optree)):
        """ return wheter or not the operation performed by optree has a local implementation """
        op_class, interface, codegen_key = key_getter(self, optree)

        if not language in op_map:
            # unsupported language
            if debug: Log.Report(Log.Info, "unsupported language for {}", optree)
            return False
        else:
            if not op_class in op_map[language]:
                # unsupported operation
                if debug: Log.report(Log.Info, "unsupported operation class for {}", optree)
                return False
            else:
                if not codegen_key in op_map[language][op_class]:
                    if debug: Log.report(Log.Info, "unsupported codegen key for {}", optree)
                    # unsupported codegen key
                    return False
                else:
                    for condition in op_map[language][op_class][codegen_key]:
                        if condition(optree):
                            for interface_condition in op_map[language][op_class][codegen_key][condition]:
                                try:
                                    if interface_condition(*interface, optree = optree): return True
                                except TypeError as e:
                                    Log.report(Log.Error, "Type Error for interface_condition on {}, {}",  op_class, (str(ifce) for ifce in interface), error=e)
                    # unsupported condition or interface type
                    if debug: 
                      Log.report(Log.Info, "unsupported condition key for {}", optree)
                      for condition in op_map[language][op_class][codegen_key]:
                          if condition(optree):
                            src_file = inspect.getsourcefile(condition)
                            _, lineno = inspect.getsourcelines(condition)
                            ic_id = 0
                            print("verified by condition @ {}:{}".format(src_file, lineno))
                            for interface_condition in op_map[language][op_class][codegen_key][condition]:
                                print("  interface_condition @{}:{}/{} {}".format(src_file, lineno, ic_id, interface_condition(*interface, optree = optree))) 
                                ic_id += 1
                      print(op_map[language][op_class][codegen_key].keys())
                      Log.report(Log.Info, "unsupported condition key for {}", optree)
                    return False

    def is_local_supported_operation(self, optree, language = C_Code, table_getter = lambda self: self.code_generation_table, debug = False,  key_getter = lambda self, optree: self.get_operation_keys(optree)):
        """ return whether or not the operation performed by optree has a local implementation """
        table = table_getter(self)
        return self.is_map_supported_operation(table, optree, language, debug = debug, key_getter = key_getter)

    def is_supported_operation(self, optree, language = C_Code, debug = False,  key_getter = lambda self, optree: self.get_operation_keys(optree)):
        """ return whether or not the operation performed by optree is supported by any level of the processor hierarchy """
        return self.is_map_supported_operation(self.simplified_rec_op_map, optree, language, debug = debug, key_getter = key_getter)

    def test_operation_support(self, op_class, out_format, in_formats, specifier=None):
        """ Test if an operation class whose prototype is out_format <- in_formats
            is supported by self target """
        # building a dummy list of inputs
        dummy_inputs = [Variable("dummy_%d" % i, precision=input_format) for i, input_format in enumerate(in_formats)]
        # build a dummy operation node
        dummy_op = op_class(*dummy_inputs, precision=out_format, specifier=specifier)
        return self.is_supported_operation(dummy_op)

    @staticmethod
    def get_operation_keys(optree):
        """ return code_generation_table key corresponding to the operation
            performed by <optree> """
        op_class = optree.__class__
        if optree.get_precision() is None:
            Log.report(
                Log.Error, "Following optree has undefined({} precision: \n{}",
                optree.get_precision(),
                optree
            )
        result_type = (optree.get_precision().get_match_format(),)
        arg_type = tuple((arg.get_precision().get_match_format() if not arg.get_precision() is None else None) for arg in optree.get_inputs())
        interface = result_type + arg_type
        codegen_key = optree.get_codegen_key()
        return op_class, interface, codegen_key


    def get_preferred_sub_vector_size(self, scalar_precision, vector_size):
        """ Returns the target preferred sub-vector size for a vector of size
            @p vector_size and of scalar precision @p scalar_precision """
        return vector_size

## Determine whether an object is a true processor
#  class with real backend capabilities or not
def test_is_processor(proc_class):
    """ return whether or not proc_class is a valid and non virtual processor class """
    return issubclass(proc_class, AbstractBackend) and not proc_class is AbstractBackend


def get_parent_proc_class_list(proc_class):
    return [parent for parent in proc_class.__bases__ if test_is_processor(parent)]
    

def create_proc_hierarchy(process_list, proc_class_list = []):
    """ create an ordered list of processor hierarchy """
    if process_list == []:
        return proc_class_list
    new_process_list = []
    for proc_class in process_list:
        if proc_class in proc_class_list: 
            continue
        else:
            proc_class_list.append(proc_class)
            new_process_list += get_parent_proc_class_list(proc_class)
    result = create_proc_hierarchy(new_process_list, proc_class_list)
    return result
    
class ML_FullySupported: pass
