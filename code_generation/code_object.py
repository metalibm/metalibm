# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013)
# All rights reserved
# created:          Dec 24th, 2013
# last-modified:    Apr  4th, 2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

import re
import commands
from pythonsollya import *

from ..core.ml_operations import Variable
from .code_constant import C_Code, Gappa_Code
from ..core.ml_formats import ML_GlobalRoundMode, ML_Fixed_Format, ML_FP_Format

from ..utility import version_info as ml_version_info


class DataLayout:
    def __init__(self):
        pass

class SymbolTable:
    def __init__(self):
        self.table = {}
        self.prefix_index = {}

    def is_free_name(self, name):
        return not name in self.table

    def get_free_name(self, var_type, prefix = "tmp"):
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

    def has_definition(self, symbol_object):
        for key in self.table:
            if symbol_object is self.table[key]: return key
        return None

    def declare_symbol(self, name, symbol_object):
        self.table[name] = symbol_object

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


class MultiSymbolTable:
    """ symbol table object """
    class ConstantSymbol: pass
    class FunctionSymbol: pass
    class VariableSymbol: pass
    class ProtectedSymbol: pass
    class TableSymbol: pass

    def get_shared_table(self, symbol_tag, shared_tables):
        if symbol_tag in shared_tables: return shared_tables[symbol_tag]
        else: return SymbolTable()

    def __init__(self, shared_tables = None, parent_tables = None): 
        """ symbol table initialization 
            shared_tables is a map of pre-defined tables shared with other parent block 
            (and not created within this block)
            parent_tables is a list of pre-existing tables, which are used when
            checking whether a name is free or not
        """
        shared_tables = shared_tables if shared_tables else {}
        parent_tables = parent_tables if parent_tables else []
        self.constant_table = self.get_shared_table(MultiSymbolTable.ConstantSymbol, shared_tables)
        self.function_table = self.get_shared_table(MultiSymbolTable.FunctionSymbol, shared_tables)
        self.variable_table = self.get_shared_table(MultiSymbolTable.VariableSymbol, shared_tables)
        self.protected_table = self.get_shared_table(MultiSymbolTable.ProtectedSymbol, shared_tables)
        self.table_table = self.get_shared_table(MultiSymbolTable.TableSymbol, shared_tables)

        self.parent_tables = parent_tables

        self.table_list = {
            MultiSymbolTable.ConstantSymbol: self.constant_table, 
            MultiSymbolTable.FunctionSymbol: self.function_table, 
            MultiSymbolTable.VariableSymbol: self.variable_table, 
            MultiSymbolTable.ProtectedSymbol: self.protected_table, 
            MultiSymbolTable.TableSymbol: self.table_table,
        }

        self.prefix_index = {}


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

    def get_free_name(self, var_type, prefix = "tmp"):
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

    def is_empty(self):
        return reduce(lambda acc, v: acc + len(v), self.table_list) == 0


    def declare_function_name(self, function_name, function_object):
        self.function_table.declare_symbol(function_name, function_object)

    def declare_var_name(self, var_name, var_object):
        self.variable_table.declare_symbol(var_name, var_object)


    def declare_cst_name(self, cst_name, cst_object):
        self.constant_table.declare_symbol(cst_name, cst_object)


    def declare_table_name(self, table_name, table_object):
        self.table_table.declare_symbol(table_name, table_object)


    def generate_declarations(self, code_generator, exclusion_list = []):
        code_object = ""
        for table_tag in self.table_list:
            if table_tag in exclusion_list:
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
    git_tag = commands.getoutput("git log -n 1")
    return git_tag


class CodeObject:
    tab = "    "
    def __init__(self, language, shared_tables = None, parent_tables = None, rounding_mode = ML_GlobalRoundMode):
        """ code object initialization """
        self.expanded_code = ""
        self.tablevel = 0
        self.header_list = []
        self.symbol_table = MultiSymbolTable(shared_tables if shared_tables else {}, parent_tables = (parent_tables if parent_tables else []))
        self.language = language
        self.header_comment = []

    def add_header_comment(self, comment):
        self.header_comment.append(comment)

    def get_symbol_table(self):
        return self.symbol_table

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



    def open_level(self):
        """ open nested block """
        self << "{\n"
        self.inc_level()

    def close_level(self, cr = "\n"):
        """ close nested block """
        self.dec_level()
        self << "}%s" % cr

    def link_level(self, transition = ""):
        """ close nested block """
        self.dec_level()
        self << "} %s {" % transition
        self.inc_level()


    def add_header(self, header_file):
        """ add a new header file """
        if not header_file in self.header_list:
            self.header_list.append(header_file)

    def generate_header_code(self, git_tag = True):
        """ generate code for header file inclusion """
        result = ""
        # generating git comment
        if git_tag:
            git_comment = "generated using metalibm %s \n sha1 git: %s \n" % (ml_version_info.version_num, ml_version_info.git_sha)
            self.header_comment.insert(0, git_comment) 
        # generating header comments
        result += "/**\n"
        for comment in self.header_comment:
            result += " * " + comment.replace("\n", "\n * ") + "\n"
        result += "**/\n"

        for header_file in self.header_list:
            result += """#include <%s>\n""" % (header_file)
        return result

    def get_free_var_name(self, var_type, prefix = "tmp", declare = True):
        free_var_name = self.symbol_table.get_free_name(var_type, prefix)
        # declare free var if required 
        if declare:
            self.symbol_table.declare_var_name(free_var_name, Variable(free_var_name, precision = var_type))

        return free_var_name

    def get_free_name(self, var_type, prefix = "tmp"):
        return self.symbol_table.get_free_name(var_type, prefix)

    def table_has_definition(self, table_object):
        return self.symbol_table.table_has_definition(table_object)


    def declare_cst(self, cst_object, prefix = "cst"):
        """ declare a new constant object and return the registered name """
        free_var_name = self.symbol_table.get_free_name(cst_object.get_precision(), prefix)
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


    def get(self, code_generator, static_cst = False, static_table = False, headers = False, skip_function = False):
        """ generate unrolled code content """
        result = ""

        if headers: 
            result += self.generate_header_code()
            result += "\n\n"

        declaration_exclusion_list = [MultiSymbolTable.ConstantSymbol] if static_cst else []
        declaration_exclusion_list += [MultiSymbolTable.TableSymbol] if static_table else []
        declaration_exclusion_list += [MultiSymbolTable.FunctionSymbol] if skip_function else []
        result += self.symbol_table.generate_declarations(code_generator, exclusion_list = declaration_exclusion_list)
        result += self.symbol_table.generate_initializations(code_generator, init_required_list = [MultiSymbolTable.ConstantSymbol, MultiSymbolTable.VariableSymbol])
        result += "\n" if result != "" else ""
        result += self.expanded_code
        return result

    def add_comment(self, comment):
        """ add a full line comment """
        self << ("/* %s */\n" % comment)

class Gappa_Unknown: 
    def __str__(self):
        return "?"


class GappaCodeObject(CodeObject):
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
        result = "# goalee\n"
        hypothesis = []
        for hc, hv in self.hypothesis_table:
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
        elif isinstance(value, SollyaObject) and PSI_is_range(value):
            return "[%s, %s]" % (inf(value), sup(value))
        else:
            return str(value)


    def get(self, code_generator, static_cst = False, static_table = False, headers = False, skip_function = True):
        result = ""

        # symbol exclusion list
        declaration_exclusion_list = [MultiSymbolTable.ConstantSymbol] if static_cst else []
        declaration_exclusion_list += [MultiSymbolTable.TableSymbol] if static_table else []
        declaration_exclusion_list += [MultiSymbolTable.VariableSymbol]
        declaration_exclusion_list += [MultiSymbolTable.FunctionSymbol] if skip_function else []

        # declaration generation
        result += self.symbol_table.generate_declarations(code_generator, exclusion_list = declaration_exclusion_list)
        result += self.symbol_table.generate_initializations(code_generator, init_required_list = [MultiSymbolTable.ConstantSymbol, MultiSymbolTable.VariableSymbol])
        result += "\n" if result != "" else ""
        result += self.expanded_code
        result += "\n\n"
        result += self.gen_complete_goal()
        result += self.gen_hint()
        return result




class NestedCode:
    """ object to support multiple levels of nested code with local and global variable management """
    def __init__(self, code_generator, static_cst = False, static_table = True):
        self.language = code_generator.language
        self.code_generator = code_generator

        self.static_cst_table = SymbolTable()
        self.static_table_table = SymbolTable()
        self.static_cst = static_cst
        self.static_table = static_table

        self.static_function_table = SymbolTable()
        
        shared_tables = {
            MultiSymbolTable.ConstantSymbol: self.get_cst_table(), 
            MultiSymbolTable.TableSymbol: self.get_table_table(),
            MultiSymbolTable.FunctionSymbol: self.get_function_table(),   
        }

        self.main_code = CodeObject(self.language, shared_tables) 
        self.code_list = [self.main_code]

    def add_header_comment(self, comment):
        self.main_code.add_header_comment(comment)

    def get_cst_table(self):
        if self.static_cst: return self.static_cst_table
        else: return SymbolTable()

    def get_table_table(self):
        if self.static_table: return self.static_table_table
        else: return SymbolTable()

    def get_function_table(self):
        return self.static_function_table
        
    def add_header(self, header_file):
        self.main_code.add_header(header_file)
        
    def __lshift__(self, added_code):
        self.code_list[0] << added_code

    def add_comment(self, comment):
        self.code_list[0].add_comment(comment)

    def open_level(self):
        self.code_list[0].open_level()
        parent_tables = self.code_list[0].get_symbol_table().get_extended_dependency_table()
        shared_tables = {
            MultiSymbolTable.ConstantSymbol: self.get_cst_table(), 
            MultiSymbolTable.TableSymbol: self.get_table_table(),
            MultiSymbolTable.FunctionSymbol: self.get_function_table(),    
        }
        self.code_list.insert(0, CodeObject(self.language, shared_tables, parent_tables = parent_tables))

    def close_level(self, cr = "\n"):
        level_code = self.code_list.pop(0)
        self << level_code.get(self.code_generator, static_cst = self.static_cst, static_table = self.static_table, skip_function = True) 
        self.code_list[0].close_level(cr = cr)

    # @param function_object possible dummy FunctionObject associated with new function_name
    def declare_free_function_name(self, prefix = "foo", function_object = None):
        function_name = self.code_list[0].get_free_name(None, prefix = prefix) 
        print "function_name: ", function_name
        self.code_list[0].declare_function(function_name, function_object)
        return function_name
      

    def get_free_var_name(self, var_type, prefix = "tmp", declare = True):
        return self.code_list[0].get_free_var_name(var_type, prefix, declare)

    def declare_cst(self, cst_object, prefix = "cst"):
        return self.code_list[0].declare_cst(cst_object, prefix)

    def declare_table(self, table_object, prefix = "table"):
        return self.code_list[0].declare_table(table_object, prefix)

    def declare_function(self, function_name, function_object):
        return self.code_list[0].declare_function(function_name, function_object)

    def get(self, code_generator, static_cst = False, static_table = False, headers = True, skip_function = False):
        return self.code_list[0].get(code_generator, static_cst = static_cst, static_table = static_table, headers = headers, skip_function = skip_function)


