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
# created:          Dec 24th, 2013
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

import re
import subprocess
import sys
import pdb

import sollya

from ..core.ml_operations import Variable
from ..core.ml_hdl_operations import Signal
from .code_constant import (
    C_Code, Gappa_Code, LLVM_IR_Code, OpenCL_Code, ASM_Code,
)
from ..core.ml_formats import ML_GlobalRoundMode, ML_Fixed_Format, ML_FP_Format, FunctionFormat

from .code_configuration import CodeConfiguration

from metalibm_core.core.meta_interval import MetaIntervalList, MetaInterval


class DataLayout(object):
    def __init__(self):
        pass

from collections import OrderedDict

class SymbolTable(object):
    def __init__(self, uniquifier=""):
        # using an ordered ditcionnary to ensure processing order
        # during code generation
        self.table = OrderedDict()
        self.reverse_map = {}
        self.prefix_index = {}
        self.uniquifier = uniquifier

    def is_free_name(self, name):
        return not name in self.table

    def uniquify(self, name):
        return self.uniquifier + name

    def is_empty(self):
        return len(self.table) == 0

    def get_free_name(self, var_type, prefix = "sttmp", update_index = True):
        _prefix = self.uniquify(prefix)
        if self.is_free_name(_prefix):
            self.prefix_index[_prefix] = 0
            return _prefix
        else:
            new_index = 0
            if _prefix in self.prefix_index:
                new_index = self.prefix_index[_prefix] + 1
            while not self.is_free_name("%s%d" % (_prefix, new_index)):
                new_index += 1
            if update_index:
              self.prefix_index[_prefix] = new_index
            return "%s%d" % (_prefix, new_index)

    def get_values(self):
        return self.table.values()

    def has_definition(self, symbol_object):
        if symbol_object in self.reverse_map:
          return self.reverse_map[symbol_object]
        else:
          return None
        #for key in self.table:
        #    if symbol_object is self.table[key]: return key
        #return None

    def declare_symbol(self, name, symbol_object):
        self.table[name] = symbol_object
        self.reverse_map[symbol_object] = name

    def generate_declaration(self, code_generator):
        code_object = ""
        for symbol in self.table:
            symbol_object = self.table[symbol]
            code_object += code_generator.generate_declaration(symbol, symbol_object)
        return code_object

    def generate_initialization(self, code_generator):
        """ generate symbol initialization, only necessary
            if symbols require a specific initialization procedure
            after declaration (e.g. mpfr_t variable) """
        code_object = ""
        for symbol in self.table:
            symbol_object = self.table[symbol]
            code_object += code_generator.generate_initialization(symbol, symbol_object)
        return code_object


class MultiSymbolTable(object):
    """ symbol table object """
    class ConstantSymbol: pass
    class FunctionSymbol: pass
    class ComponentSymbol: pass
    class EntitySymbol: pass
    class VariableSymbol: pass
    class SignalSymbol:  pass
    class ProtectedSymbol: pass
    class TableSymbol: pass
    class LabelSymbol: pass

    def get_shared_table(self, symbol_tag, shared_tables):
        if symbol_tag in shared_tables: return shared_tables[symbol_tag]
        else: return SymbolTable(uniquifier = self.uniquifier)

    def __init__(self, shared_tables = None, parent_tables = None, uniquifier = ""):
        """ symbol table initialization
            shared_tables is a map of pre-defined tables shared with other parent block
            (and not created within this block)
            parent_tables is a list of pre-existing tables, which are used when
            checking whether a name is free or not
        """
        self.uniquifier = uniquifier

        self.default_exclusion_list = [MultiSymbolTable.ProtectedSymbol]

        shared_tables = shared_tables if shared_tables else {}
        parent_tables = parent_tables if parent_tables else []
        self.constant_table = self.get_shared_table(MultiSymbolTable.ConstantSymbol, shared_tables)
        self.function_table = self.get_shared_table(MultiSymbolTable.FunctionSymbol, shared_tables)
        self.variable_table = self.get_shared_table(MultiSymbolTable.VariableSymbol, shared_tables)
        self.signal_table   = self.get_shared_table(MultiSymbolTable.SignalSymbol, shared_tables)
        self.protected_table = self.get_shared_table(MultiSymbolTable.ProtectedSymbol, shared_tables)
        self.table_table = self.get_shared_table(MultiSymbolTable.TableSymbol, shared_tables)
        self.component_table = self.get_shared_table(MultiSymbolTable.ComponentSymbol, shared_tables)
        self.entity_table    = self.get_shared_table(MultiSymbolTable.EntitySymbol, shared_tables)
        self.label_table = self.get_shared_table(MultiSymbolTable.LabelSymbol, shared_tables)

        self.parent_tables = parent_tables

        # initializing through a list of paire (key, value) to ensure
        # known ordering
        self.table_list = OrderedDict([
            (MultiSymbolTable.ConstantSymbol, self.constant_table),
            (MultiSymbolTable.FunctionSymbol, self.function_table),
            (MultiSymbolTable.VariableSymbol, self.variable_table),
            (MultiSymbolTable.SignalSymbol,   self.signal_table),
            (MultiSymbolTable.ProtectedSymbol, self.protected_table),
            (MultiSymbolTable.TableSymbol, self.table_table),
            (MultiSymbolTable.ComponentSymbol, self.component_table),
            (MultiSymbolTable.EntitySymbol, self.entity_table),
            (MultiSymbolTable.LabelSymbol, self.label_table),
        ])

        self.prefix_index = {}

    def is_empty(self):
      for table_tag in self.table_list:
        if not self.table_list[table_tag].is_empty(): return False
      return True

    def table_has_definition(self, table_object):
        """ search for a previous definition of ML_Table <table_object>
            returns the table index if found, else None """
        table_key = self.table_table.has_definition(table_object)
        if table_key != None:
            return table_key
        for table in self.parent_tables:
            table_name = table.table_has_definition(table_object)
            if table_name != None: return table_name
        return None

    def get_table(self, symbol):
        return self.table_list[symbol]

    def get_parent_tables(self):
        return self.parent_tables

    def get_extended_dependency_table(self):
        return self.parent_tables + [self]

    def is_free_name(self, name):
        for table_tag in self.table_list:
            if not self.table_list[table_tag].is_free_name(name): return False
        for table in self.parent_tables:
            if not table.is_free_name(name): return False
        return True

    def get_free_name(self, var_type, prefix = "mstmp"):
        if self.is_free_name(prefix):
            self.prefix_index[prefix] = 0
            return prefix
        else:
            new_index = 0
            if prefix in self.prefix_index:
                new_index = self.prefix_index[prefix] + 1
            while not self.is_free_name("%s%d" % (prefix, new_index)):
                new_index += 1
            self.prefix_index[prefix] = new_index
            return "%s%d" % (prefix, new_index)

    ## return a free name for a constant
    def get_free_cst_name(self, var_type, prefix = "cst"):
        cst_free_name = self.constant_table.get_free_name(var_type, prefix, update_index = True)
        while not self.is_free_name(cst_free_name):
          cst_free_name = self.constant_table.get_free_name(var_type, prefix, update_index = True)
        return cst_free_name


    def declare_symbol_name(self, symbol_name, symbol_object, symbol_type):
        """ Generic function to declare a symbol of a specified type """
        self.table_list[symbol_type].declare_symbol(symbol_name, symbol_object)


    def declare_function_name(self, function_name, function_object):
        self.function_table.declare_symbol(function_name, function_object)

    def declare_component_name(self, component_name, component_object):
        self.component_table.declare_symbol(component_name, component_object)
        self.entity_table.declare_symbol(component_name, component_object)

    def get_entity_list(self):
        """ return a list of components registered in this table """
        return self.entity_table.get_values()

    def declare_protected_symbol(self, name, node):
        self.protected_table.declare_symbol(name, node)

    def declare_var_name(self, var_name, var_object):
        self.variable_table.declare_symbol(var_name, var_object)

    def declare_signal_name(self, signal_name, signal_object):
        self.signal_table.declare_symbol(signal_name, signal_object)

    def has_signal_definition(self, signal_object):
        return self.signal_table.has_definition(signal_object)

    def declare_cst_name(self, cst_name, cst_object):
        self.constant_table.declare_symbol(cst_name, cst_object)


    def declare_table_name(self, table_name, table_object):
        self.table_table.declare_symbol(table_name, table_object)


    def generate_declarations(self, code_generator, exclusion_list = []):
        code_object = ""
        for table_tag in self.table_list:
            if table_tag in exclusion_list + self.default_exclusion_list:
                continue
            code_object += self.table_list[table_tag].generate_declaration(code_generator)
        return code_object

    def generate_initializations(self, code_generator, init_required_list = []):
        code_object = ""
        for table_tag in init_required_list:
            code_object += self.table_list[table_tag].generate_initialization(code_generator)
        return code_object


def get_git_tag():
    """ extract git commit tag """
    git_tag = subprocess.call("git log -n 1")
    return git_tag

class CodeObjectRegistry:
    code_object_class_map = {}

def RegisterDefaultCodeObject(language_list):
    """ generate a class decorator to register
        default code object for a given language list """
    def __register(CodeObjectClass):
        for language in language_list:
            if language in CodeObjectRegistry.code_object_class_map:
                if CodeObjectRegistry.code_object_class_map[language] == CodeObjectClass:
                    Log.report(Log.Warning, "multiple registration of codegenerator {} for {}",
                               CodeObjectClass, language)
                else:
                    Log.report(Log.Warning, "language {} has an already registered code generator class {} (when trying to register {})", 
                               language, CodeObjectClass.code_object_class_map[language], CodeObjectClass)
            CodeObjectRegistry.code_object_class_map[language] = CodeObjectClass
        return CodeObjectClass
    return __register

class CommonCodeObject:
    """ common methods for RTL and C-like code object classes """
    def add_header_comment(self, comment):
        self.header_comment.append(comment)

    def is_empty(self):
        return len(self.header_list) == 0 and len(self.library_list) == 0 and self.symbol_table.is_empty() and len(self.header_comment) == 0 and len(self.expanded_code) == 0

    def get_symbol_table(self):
        return self.symbol_table

    def add_header(self, header_file):
        """ add a new header file """
        if not header_file in self.header_list:
            self.header_list.append(header_file)

    def add_library(self, library_file):
        """ add a new library file """
        if not library_file in self.library_list:
            self.library_list.append(library_file)

    def register_object_type_header(self, symbol_object):
        symbol_type = symbol_object.get_precision()
        if not symbol_type.builtin_type:
            self.add_header(symbol_type.header)

    def table_has_definition(self, table_object):
        return self.symbol_table.table_has_definition(table_object)




class CodeObject(CodeConfiguration, CommonCodeObject):
    level_header = "{\n"
    level_footer = "}"
    # list of symbol type to be excluded from declaration during
    # get / push_into_parent_code
    DEFAULT_EXCLUSION_LIST = []
    # Prefix added to every free variable name
    GENERAL_PREFIX = ""
    def __init__(self, language, shared_tables = None, parent_tables = None, rounding_mode = ML_GlobalRoundMode, uniquifier = "", main_code_level = None, var_ctor = None):
        """ code object initialization """
        self.expanded_code = ""
        self.uniquifier = uniquifier
        self.tablevel = 0
        # list of header files which must be included before source code
        self.header_list = []
        # list of libraries required to link a binary built from this source code
        self.library_list = []
        self.symbol_table = MultiSymbolTable(shared_tables if shared_tables else {}, parent_tables = (parent_tables if parent_tables else []), uniquifier = self.uniquifier)
        self.language = language
        self.header_comment = []
        self.default_var_ctor = var_ctor

    def reindent(self, line):
        """ re indent code line <line> with proper current indentation level """
        # inserting proper indentation level
        codeline = re.sub("\n", lambda _: ("\n" + self.tablevel * CodeObject.tab), line)
        # removing trailing whitespaces
        codeline = re.sub(" +\n", "\n", codeline)
        return codeline

    def append_code(self, code):
        self.expanded_code += code
        return self

    def __lshift__(self, added_code):
        """ implicit code insertion through << operator """
        indented_code = self.reindent(added_code)
        return self.append_code(indented_code)

    def inc_level(self):
        """ increase indentation level """
        self.tablevel += 1
        self.expanded_code += CodeObject.tab

    def dec_level(self):
        """ decrease indentation level """
        self.tablevel -= 1
        # deleting last inserted tab
        if self.expanded_code[-len(CodeObject.tab):] == CodeObject.tab:
            self.expanded_code = self.expanded_code[:-len(CodeObject.tab)]

    def open_level(self, inc=True, header=None):
        """ open nested block """
        header = self.level_header if header is None else header
        self << header # "{\n"
        if inc: self.inc_level()

    def close_level(self, cr="\n", inc=True, footer=None):
        """ close nested block """
        footer = self.level_footer if footer is None else footer
        if inc: self.dec_level()
        self << "%s%s" % (footer, cr)

    def link_level(self, transition = ""):
        """ close nested block """
        self.dec_level()
        self << "} %s {" % transition
        self.inc_level()

    def get_multiline_comment(self, comment_list):
        result = "/**"
        def empty_line(s):
            return not len(s.replace(" ", "").replace("\t", ""))
        for comment in comment_list:
            # result += " * " + comment.replace("\n", "\n * ") + "\n"
            result += "\n" + "\n".join(map(lambda s: (" *" if empty_line(s) else " * " + s), comment.split("\n")))
        result += "\n */\n"
        return result

    def generate_header_code(self, git_tag = True):
        """ generate code for header file inclusion """
        result = ""

        # generating git comment
        if git_tag:
            git_comment = CodeConfiguration.get_git_comment()
            self.header_comment.insert(0, git_comment)
        # generating header comments
        result += self.get_multiline_comment(self.header_comment)

        for header_file in self.header_list:
            result += """#include <%s>\n""" % (header_file)
        return result

    def get_free_symbol_name(self, symbol_type, symbol_format, prefix="cotmp", declare=True, symbol_ctor=Variable):
        free_symbol_name = self.symbol_table.get_free_name(symbol_format, prefix)
        if declare:
            self.symbol_table.declare_symbol_name(free_symbol_name, symbol_ctor(free_symbol_name), symbol_type=symbol_type)
        return free_symbol_name

    def get_free_var_name(self, var_type, prefix = "cotmp", declare = True, var_ctor = Variable):
        assert not var_type is None
        free_var_name = self.symbol_table.get_free_name(var_type, self.GENERAL_PREFIX + prefix)
        # declare free var if required
        if declare:
            self.symbol_table.declare_var_name(free_var_name, var_ctor(free_var_name, precision = var_type))
        return free_var_name

    def declare_var_name(self, var_name, var_object):
        self.symbol_table.declare_var_name(var_name, var_object)
        return var_name

    def declare_protected_symbol(self, name, node):
        self.symbol_table.declare_protected_symbol(name, node)
        return node

    def get_free_name(self, var_type, prefix = "cotmp"):
        assert not var_type is None
        return self.symbol_table.get_free_name(var_type, prefix)

    ## Declare a new constant object whose name is build
    #  from @p prefix
    #  @param cst_objet Constant constant object to be declared
    #  @para, prefix str constant name prefix
    def declare_cst(self, cst_object, prefix = "cst"):
        """ declare a new constant object and return the registered name """
        free_var_name = self.symbol_table.get_free_cst_name(cst_object.get_precision(), prefix)
        self.symbol_table.declare_cst_name(free_var_name, cst_object)
        return free_var_name

    def declare_table(self, table_object, prefix):
        table_name = self.table_has_definition(table_object)
        if table_name != None:
            return table_name
        else:
            free_var_name = self.symbol_table.get_free_name(table_object.get_storage_precision(), prefix)
            self.symbol_table.declare_table_name(free_var_name, table_object)
            return free_var_name

    def declare_function(self, function_name, function_object):
        self.symbol_table.declare_function_name(function_name, function_object)
        return function_name

    def declare_component(self, component_name, component_object):
        self.symbol_table.declare_component_name(component_name, component_object)
        return component_name


    def get(self, code_generator, static_cst = False, static_table = False, headers = False, skip_function = False):
        """ generate unrolled code content """
        result = ""

        if headers:
            result += self.generate_header_code()
            result += "\n\n"

        declaration_exclusion_list = [MultiSymbolTable.ConstantSymbol] if static_cst else []
        declaration_exclusion_list += [MultiSymbolTable.TableSymbol] if static_table else []
        declaration_exclusion_list += [MultiSymbolTable.FunctionSymbol] if skip_function else []
        declaration_exclusion_list += self.DEFAULT_EXCLUSION_LIST
        result += self.symbol_table.generate_declarations(code_generator, exclusion_list = declaration_exclusion_list)
        result += self.symbol_table.generate_initializations(code_generator, init_required_list = [MultiSymbolTable.ConstantSymbol, MultiSymbolTable.VariableSymbol])
        result += "\n" if result != "" else ""
        result += self.expanded_code
        return result

    def push_into_parent_code(self, parent_code, code_generator, static_cst = False, static_table = False, headers = False, skip_function = False):
        if headers:
            parent_code << self.generate_header_code()
            parent_code << "\n\n"

        declaration_exclusion_list = [MultiSymbolTable.ConstantSymbol] if static_cst else []
        declaration_exclusion_list += [MultiSymbolTable.TableSymbol] if static_table else []
        declaration_exclusion_list += [MultiSymbolTable.FunctionSymbol] if skip_function else []
        declaration_exclusion_list += self.DEFAULT_EXCLUSION_LIST
        parent_code << self.symbol_table.generate_declarations(code_generator, exclusion_list = declaration_exclusion_list)
        parent_code << self.symbol_table.generate_initializations(code_generator, init_required_list = [MultiSymbolTable.ConstantSymbol, MultiSymbolTable.VariableSymbol])
        parent_code << "\n"
        parent_code << self.expanded_code

    def add_comment(self, comment):
        """ add a full line comment """
        self << ("/* %s */\n" % comment)
    def add_multiline_comment(self, comment):
        self.add_comment(comment)

@RegisterDefaultCodeObject([C_Code, OpenCL_Code])
class CstyleCodeObject(CodeObject):
    pass

@RegisterDefaultCodeObject([ASM_Code])
class AsmCodeObject(CodeObject):
    # function/level opening does not have a specific symbol
    level_header = ""
    # function/level closing does not have a specific symbol
    level_footer = ""

    # always exclude the following from asm code generation
    DEFAULT_EXCLUSION_LIST = [
        MultiSymbolTable.VariableSymbol,
        MultiSymbolTable.LabelSymbol
    ]

class Gappa_Unknown(object):
    def __str__(self):
        return "?"


class GappaCodeObject(CodeObject):
    DEFAULT_EXCLUSION_LIST = [MultiSymbolTable.VariableSymbol]
    def __init__(self):
        CodeObject.__init__(self, Gappa_Code)
        self.hint_table = []
        self.hypothesis_table = []
        self.goal_table = []

    def add_hint(self, hypoth_code, goal_code, annotation_code, isApprox = False):
        self.hint_table.append((hypoth_code, goal_code, annotation_code, isApprox))

    def add_hypothesis(self, hypoth_code, hypoth_value):
        self.hypothesis_table.append((hypoth_code, hypoth_value))

    def add_goal(self, goal_code, goal_value = Gappa_Unknown):
        self.goal_table.append((goal_code, goal_value))

    def gen_hint(self):
        result = "#hints\n"
        for hypoth_code, goal_code, annotation_code, isApprox in self.hint_table:
            annotation_code = "{%s}" % annotation_code.get() if annotation_code is not None else ""
            symbol = "~" if isApprox else "->"
            result += "%s %s %s %s;\n\n" % (hypoth_code.get(), symbol, goal_code.get(), annotation_code)
        return result

    def gen_complete_goal(self):
        result = "# goals\n"
        hypothesis = []
        for hc, hv in self.hypothesis_table:
            if isinstance(hv, MetaIntervalList):
                disjonction_list =[
                    ("%s in %s" % (hc.get(), self.get_value_str(sub_interval))) for sub_interval in hv.interval_list]
                disjonction = " \/ ".join(disjonction_list)
                hypothesis.append("( %s )" % disjonction)
            else:
                hypothesis.append("%s in %s" % (hc.get(), self.get_value_str(hv)))
            if isinstance(hc.precision, ML_Fixed_Format):
                hypothesis.append("@FIX(%s,%s)" % (hc.get(), str(- hc.precision.get_frac_size())))
            if isinstance(hc.precision, ML_FP_Format):
                hypothesis.append("@FLT(%s,%s)" % (hc.get(), str(hc.precision.get_field_size()+1)))
        goal = ["%s in %s" % (hc.get(), self.get_value_str(hv)) for hc, hv in self.goal_table]
        result += "{ %s -> %s }\n\n" % (" /\ ".join(hypothesis), " /\ ".join(goal))
        return result


    def get_value_str(self, value):
        if value is Gappa_Unknown:
            return "?"
        elif isinstance(value, MetaInterval):
            return self.get_value_str(value.interval)
        elif isinstance(value, MetaIntervalList):
            # MetaIntervalList should have been catched early and 
            # should have generated a disjonction of cases
            raise NotImplementedError
        elif isinstance(value, sollya.SollyaObject) and value.is_range():
            return "[%s, %s]" % (sollya.inf(value), sollya.sup(value))
        else:
            return str(value)


    def get(self, code_generator, static_cst = False, static_table = False, headers = False, skip_function = True):
        result = ""

        # symbol exclusion list
        declaration_exclusion_list = [MultiSymbolTable.ConstantSymbol] if static_cst else []
        declaration_exclusion_list += [MultiSymbolTable.TableSymbol] if static_table else []
        declaration_exclusion_list += [MultiSymbolTable.FunctionSymbol] if skip_function else []
        declaration_exclusion_list += self.DEFAULT_EXCLUSION_LIST

        # declaration generation
        result += self.symbol_table.generate_declarations(code_generator, exclusion_list = declaration_exclusion_list)
        result += self.symbol_table.generate_initializations(code_generator, init_required_list = [MultiSymbolTable.ConstantSymbol, MultiSymbolTable.VariableSymbol])
        result += "\n" if result != "" else ""
        result += self.expanded_code
        result += "\n\n"
        result += self.gen_complete_goal()
        result += self.gen_hint()
        return result

    def push_into_parent_code(self, parent_code, code_generator, static_cst = False, static_table = False, headers = False, skip_function = False):
        # symbol exclusion list
        declaration_exclusion_list = [MultiSymbolTable.ConstantSymbol] if static_cst else []
        declaration_exclusion_list += [MultiSymbolTable.TableSymbol] if static_table else []
        declaration_exclusion_list += [MultiSymbolTable.FunctionSymbol] if skip_function else []
        declaration_exclusion_list += self.DEFAULT_EXCLUSION_LIST

        # declaration generation
        parent_code << self.symbol_table.generate_declarations(code_generator, exclusion_list = declaration_exclusion_list)
        parent_code << self.symbol_table.generate_initializations(code_generator, init_required_list = [MultiSymbolTable.ConstantSymbol, MultiSymbolTable.VariableSymbol])
        parent_code << "\n"
        parent_code << self.expanded_code
        parent_code << "\n\n"
        parent_code << self.gen_complete_goal()
        parent_code << self.gen_hint()

@RegisterDefaultCodeObject([LLVM_IR_Code])
class LLVMCodeObject(CodeObject):
    GENERAL_PREFIX = "%"
    # always exclude the following from llvm-ir code generation
    DEFAULT_EXCLUSION_LIST = [
        MultiSymbolTable.VariableSymbol,
        MultiSymbolTable.LabelSymbol
    ]

    def __init__(self, language, shared_tables=None, parent_tables=None,
                 rounding_mode=ML_GlobalRoundMode, uniquifier="",
                 main_code_level=None, var_ctor=None):
        CodeObject.__init__(
            self, LLVM_IR_Code, shared_tables, parent_tables, rounding_mode,
            uniquifier, main_code_level, var_ctor
        )


    def get_multiline_comment(self, comment_list):
        result = ""
        for comment in comment_list:
            result += "; " + comment.replace("\n", "\n;") + "\n"
        return result

    def get(self, code_generator, static_cst=False, static_table=False, headers=False, skip_function = True):
        result = ""

        if headers:
            result += self.generate_header_code()
            result += "\n\n"

        # symbol exclusion list
        declaration_exclusion_list = [MultiSymbolTable.ConstantSymbol] if static_cst else []
        declaration_exclusion_list += [MultiSymbolTable.TableSymbol] if static_table else []
        declaration_exclusion_list += [MultiSymbolTable.FunctionSymbol] if skip_function else []
        declaration_exclusion_list += [MultiSymbolTable.VariableSymbol]
        declaration_exclusion_list += self.DEFAULT_EXCLUSION_LIST

        # declaration generation
        result += self.symbol_table.generate_declarations(code_generator, exclusion_list=declaration_exclusion_list)
        result += self.symbol_table.generate_initializations(
            code_generator,
            init_required_list=[
                MultiSymbolTable.ConstantSymbol, MultiSymbolTable.VariableSymbol
            ]
        )
        result += "\n" if result != "" else ""
        result += self.expanded_code
        result += "\n"
        return result

    def push_into_parent_code(self, parent_code, code_generator, static_cst = False, static_table = False, headers = False, skip_function=True):
        # symbol exclusion list
        declaration_exclusion_list = [MultiSymbolTable.ConstantSymbol] if static_cst else []
        declaration_exclusion_list += [MultiSymbolTable.TableSymbol] if static_table else []
        declaration_exclusion_list += [MultiSymbolTable.FunctionSymbol] if skip_function else []
        declaration_exclusion_list += self.DEFAULT_EXCLUSION_LIST

        # declaration generation
        parent_code << self.symbol_table.generate_declarations(code_generator, exclusion_list = declaration_exclusion_list)
        parent_code << self.symbol_table.generate_initializations(code_generator, init_required_list = [MultiSymbolTable.ConstantSymbol, MultiSymbolTable.VariableSymbol])
        parent_code << "\n"
        parent_code << self.expanded_code
        parent_code << "\n"


class VHDLCodeObject(CodeConfiguration, CommonCodeObject):
    def __init__(self, language, shared_tables = None, parent_tables = None, rounding_mode = ML_GlobalRoundMode, uniquifier = "", main_code_level = False, var_ctor = None):
        """ code object initialization """
        self.expanded_code = ""
        self.uniquifier = uniquifier
        self.tablevel = 0
        self.header_list = []
        self.library_list = []
        self.symbol_table = MultiSymbolTable(shared_tables if shared_tables else {}, parent_tables = (parent_tables if parent_tables else []), uniquifier = self.uniquifier)
        self.language = language
        self.header_comment = []
        self.shared_symbol_table_f = MultiSymbolTable.SignalSymbol in shared_tables
        self.main_code_level = main_code_level
        self.default_var_ctor = var_ctor

    def __lshift__(self, added_code):
        """ implicit code insertion through << operator """
        self.expanded_code += re.sub("\n", lambda _: ("\n" + self.tablevel * CodeObject.tab), added_code)

    def inc_level(self):
        """ increase indentation level """
        self.tablevel += 1
        self.expanded_code += CodeObject.tab

    def dec_level(self):
        """ decrease indentation level """
        self.tablevel -= 1
        # deleting last inserted tab
        if self.expanded_code[-len(CodeObject.tab):] == CodeObject.tab:
            self.expanded_code = self.expanded_code[:-len(CodeObject.tab)]

    def open_level(self, inc = True, header=None):
        """ open nested block """
        if inc: self.inc_level()

    def close_level(self, cr = "\n", inc = True, footer=None):
        """ close nested block """
        if inc: self.dec_level()

    def link_level(self, transition = ""):
        """ close nested block """
        raise NotImplementedError

    def generate_header_code(self, git_tag = True):
        """ generate code for header file inclusion """
        result = ""
        # generating git comment
        if git_tag:
            git_comment = CodeConfiguration.get_git_comment()
            self.header_comment.insert(0, git_comment)

        # generating header comments
        result += "--\n"
        for comment in self.header_comment:
            result += "-- " + comment.replace("\n", "\n--") + "\n"
        # TODO/FIXME: erase trailing white spaces properly
        result.replace("--\n", "--\n")
        result += "--\n"

        for library_file in self.library_list:
            result += "library {lib};\n".format(lib = library_file)

        for header_file in self.header_list:
            result += """use %s;\n""" % (header_file)
        return result

    def get_free_var_name(self, var_type, prefix = "tmps", declare = True, var_ctor = None):
        assert not var_type is None
        var_ctor = var_ctor or self.default_var_ctor or Signal
        free_var_name = self.symbol_table.get_free_name(var_type, prefix)
        # declare free var if required
        if declare:
            if var_ctor is Variable:
                self.symbol_table.declare_var_name(free_var_name, var_ctor(free_var_name, precision = var_type))
            elif var_ctor is Signal:
                self.symbol_table.declare_signal_name(free_var_name, var_ctor(free_var_name, precision = var_type))
            else:
                Log.report(Log.Error, "unsupported var constructor in get_free_var_name")
        return free_var_name

    def get_free_signal_name(self, signal_type, prefix = "stmp", declare = True):
        assert not signal_type is None
        free_signal_name = self.symbol_table.get_free_name(signal_type, prefix)
        # declare free var if required
        if declare:
            self.symbol_table.declare_signal_name(free_signal_name, Signal(free_signal_name, precision = signal_type))
        return free_signal_name

    def declare_var_name(self, var_name, var_object):
        self.symbol_table.declare_var_name(var_name, var_object)
        return var_name

    def declare_protected_symbol(self, name, node):
        self.symbol_table.declare_protected_symbol(name, node)
        return node

    def declare_signal(self, signal_object, signal_type, prefix = "stmp"):
        signal_key = self.symbol_table.has_signal_definition(signal_object)
        if not signal_key is None:
          return signal_key
        else:
          free_signal_name = self.symbol_table.get_free_name(signal_type, prefix)
          self.symbol_table.declare_signal_name(free_signal_name, signal_object)
        return free_signal_name

    def get_free_name(self, var_type, prefix = "svtmp"):
        return self.symbol_table.get_free_name(var_type, prefix)

    ## Declare a new constant object whose name is build
    #  from @p prefix
    #  @param cst_objet Constant constant object to be declared
    #  @para, prefix str constant name prefix
    def declare_cst(self, cst_object, prefix = "cst"):
        """ declare a new constant object and return the registered name """
        free_var_name = self.symbol_table.get_free_cst_name(cst_object.get_precision(), prefix)
        self.symbol_table.declare_cst_name(free_var_name, cst_object)
        return free_var_name

    def declare_table(self, table_object, prefix):
        table_name = self.table_has_definition(table_object)
        if table_name != None:
            return table_name
        else:
            free_var_name = self.symbol_table.get_free_name(table_object.get_storage_precision(), prefix)
            self.symbol_table.declare_table_name(free_var_name, table_object)
            return free_var_name


    def declare_function(self, function_name, function_object):
        self.symbol_table.declare_function_name(function_name, function_object)
        return function_name

    def declare_component(self, component_name, component_object):
        self.symbol_table.declare_component_name(component_name, component_object)
        return component_name

    def get_entity_list(self):
        """ Return the list of component instanciated in this code object """
        return self.symbol_table.get_entity_list()


    def get(self, code_generator, static_cst = False, static_table = False, headers = False, skip_function = False):
        """ generate unrolled code content """
        result = ""

        if headers:
            result += self.generate_header_code()
            result += "\n\n"

        # Entities are always excluded from generation
        # They will be processed in a separate manner (see ML_EntityBasis)
        declaration_exclusion_list = [MultiSymbolTable.EntitySymbol]
        if static_cst:
            declaration_exclusion_list.append(MultiSymbolTable.ConstantSymbol)
        if static_table:
            declaration_exclusion_list.append(MultiSymbolTable.TableSymbol)
        if skip_function:
            declaration_exclusion_list.append(MultiSymbolTable.FunctionSymbol)
        if self.shared_symbol_table_f:
            declaration_exclusion_list.append(MultiSymbolTable.SignalSymbol)

        result += self.symbol_table.generate_declarations(code_generator, exclusion_list = declaration_exclusion_list)
        result += self.symbol_table.generate_initializations(code_generator, init_required_list = [MultiSymbolTable.ConstantSymbol, MultiSymbolTable.VariableSymbol])
        result += "begin\n" if not self.main_code_level else ""
        result += "\n" if result != "" else ""
        result += self.expanded_code
        return result

    def push_into_parent_code(self, parent_code, code_generator, static_cst = False, static_table = False, headers = False, skip_function = False):
        """ generate unrolled code content """

        if headers:
            parent_code << self.generate_header_code()
            parent_code << "\n\n"

        declaration_exclusion_list = [MultiSymbolTable.EntitySymbol]
        if static_cst:
            declaration_exclusion_list.append(MultiSymbolTable.ConstantSymbol)
        if static_table:
            declaration_exclusion_list.append(MultiSymbolTable.TableSymbol)
        if skip_function:
            declaration_exclusion_list.append(MultiSymbolTable.FunctionSymbol)
        if self.shared_symbol_table_f:
            declaration_exclusion_list.append(MultiSymbolTable.SignalSymbol)
        parent_code << self.symbol_table.generate_declarations(code_generator, exclusion_list = declaration_exclusion_list)
        parent_code << self.symbol_table.generate_initializations(code_generator, init_required_list = [MultiSymbolTable.ConstantSymbol, MultiSymbolTable.VariableSymbol])
        parent_code.dec_level()
        parent_code << "\n"
        parent_code << ("begin\n" if not self.main_code_level else "")
        parent_code.inc_level()
        parent_code << self.expanded_code

    def add_comment(self, comment):
        """ add a full line comment """
        self << ("-- %s\n" % comment)

    def add_multiline_comment(self, comment):
        for line in comment.split("\n"):
            if line != "": self.add_comment(line) 

## Nested code object
#  language is derived from code_generator's language
class NestedCode(object):
    """ object to support multiple levels of nested code with local and
        global variable management """
    ##
    #  @param uniquifier <str> unifiquation prefix for name generation
    def __init__(self, code_generator,
            static_cst=False,
            static_table=True,
            uniquifier="",
            code_ctor=CodeObject,
            main_code_level=None,
            shared_symbol_list=None,
            static_var=False):
        self.language = code_generator.language
        self.code_generator = code_generator
        # name uniquifier
        self.uniquifier = uniquifier
        # constructor function for code levels
        self.code_ctor = code_ctor

        # top-level global tables
        self.global_tables = {}
        # defaulting list of shared symbol table to build
        # if none is defined
        self.global_symbol_list = [
            MultiSymbolTable.ConstantSymbol,
            MultiSymbolTable.TableSymbol,
            MultiSymbolTable.FunctionSymbol,
            MultiSymbolTable.EntitySymbol,
            MultiSymbolTable.ProtectedSymbol,
        ] if shared_symbol_list is None else shared_symbol_list

        for symbol in self.global_symbol_list:
            self.global_tables[symbol] = SymbolTable(uniquifier=uniquifier)

        shared_tables = self.global_tables

        self.main_code = self.code_ctor(self.language, shared_tables, uniquifier=self.uniquifier, main_code_level=True)
        self.code_list = [self.main_code]

    @property
    def default_var_ctor(self):
        return self.code_list[0].default_var_ctor

    def add_header_comment(self, comment):
        self.main_code.add_header_comment(comment)

    def add_header(self, header_file):
        self.main_code.add_header(header_file)

    def add_library(self, library_file):
        self.main_code.add_library(library_file)

    def add_local_header(self, header_file):
        self.code_list[0].add_header(header_file)

    def append_code(self, code):
        self.code_list[0].append_code(code)
        return self

    def __lshift__(self, added_code):
        self.code_list[0] << added_code
        return self

    def add_comment(self, comment):
        self.code_list[0].add_comment(comment)
    def add_multiline_comment(self, comment):
        self.code_list[0].add_multiline_comment(comment)

    def open_level(self, extra_shared_tables=None, inc=True, var_ctor=None, header=None):
        """ Open a new level of code """
        self.code_list[0].open_level(inc=inc, header=header)
        parent_tables = self.code_list[0].get_symbol_table().get_extended_dependency_table()

        shared_tables = {}
        # preparing sub-global shared tables
        for global_symbol in self.global_tables:
            shared_tables[global_symbol] = self.global_tables[global_symbol]
        if extra_shared_tables:
          for table_key in extra_shared_tables:
            shared_tables[table_key] = self.code_list[0].get_symbol_table().get_table(table_key)
        # building a new code object for the sub-level
        sublevel_code = self.code_ctor(
            self.language, shared_tables,
            parent_tables=parent_tables, var_ctor=var_ctor)
        self.code_list.insert(0, sublevel_code)

    @property
    def static_cst(self):
        return MultiSymbolTable.ConstantSymbol in self.global_tables
    @property
    def static_table(self):
        return MultiSymbolTable.TableSymbol in self.global_tables

    def close_level(self, cr = "\n", inc = True, footer=None):
        level_code = self.code_list.pop(0)
        level_code.push_into_parent_code(self, self.code_generator, static_cst = self.static_cst, static_table = self.static_table, skip_function = True)
        self.code_list[0].close_level(cr=cr, inc=inc, footer=footer)

    def inc_level(self):
        """ increase indentation level """
        self.code_list[0].inc_level()

    def dec_level(self):
        """ decrease indentation level """
        self.code_list[0].dec_level()

    # @param function_object possible dummy FunctionObject associated with new function_name
    def declare_free_function_name(self, prefix = "foo", function_object = None):
        function_name = self.code_list[0].get_free_name(FunctionFormat(), prefix = prefix)
        self.code_list[0].declare_function(function_name, function_object)
        return function_name

    def get_free_symbol_name(self, symbol_type, symbol_format, symbol_ctor, prefix="tmp", declare=True):
        return self.code_list[0].get_free_symbol_name(symbol_type, symbol_format, prefix, declare, symbol_ctor=symbol_ctor)


    def get_free_var_name(self, var_type, prefix = "tmpv", declare = True, var_ctor = None):
        # trying not to override code_list[0] default var_ctor
        if var_ctor is None:
          return self.code_list[0].get_free_var_name(var_type, prefix, declare)
        else:
          return self.code_list[0].get_free_var_name(var_type, prefix, declare, var_ctor)

    def get_free_name(self, var_type, prefix="tmps"):
        return self.code_list[0].get_free_name(var_type, prefix)

    def get_free_signal_name(self, signal_type, prefix = "stmp", declare = True):
        return self.code_list[0].get_free_signal_name(signal_type, prefix, declare)

    def declare_signal(self, signal_object, signal_type, prefix = "stmp"):
        return self.code_list[0].declare_signal(signal_object, signal_type, prefix)

    def declare_var_name(self, var_name, var_object):
        self.main_code.register_object_type_header(var_object)
        return self.code_list[0].declare_var_name(var_name, var_object)
    def declare_protected_symbol(self, name, node):
        return self.code_list[0].declare_protected_symbol(name, node)

    def declare_cst(self, cst_object, prefix = "cst"):
        self.main_code.register_object_type_header(cst_object)
        return self.code_list[0].declare_cst(cst_object, prefix)

    def declare_table(self, table_object, prefix = "table"):
        self.main_code.register_object_type_header(table_object)
        return self.code_list[0].declare_table(table_object, prefix)

    def declare_function(self, function_name, function_object):
        return self.code_list[0].declare_function(function_name, function_object)

    def declare_component(self, component_name, component_object):
        component_name = self.code_list[0].declare_component(component_name, component_object)
        return component_name

    def get_entity_list(self):
        """ Return the component list """
        return self.code_list[0].get_entity_list()

    def get(self, code_generator, static_cst = False, static_table = False, headers = True, skip_function = False):
        return self.code_list[0].get(code_generator, static_cst = static_cst, static_table = static_table, headers = headers, skip_function = skip_function)

    def push_into_parent_code(self, parent_code, code_generator, static_cst = False, static_table = False, headers = False, skip_function = False):
        return self.code_list[0].push_into_parent_code(parent_code, code_generator, static_cst, static_table, headers, skip_function)


