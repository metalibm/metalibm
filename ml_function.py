from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.code_element import CodeFunction
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.utility.log_report import Log



class ML_Function:
    """A class from which all metafunction inherit"""
    def __init__(self,
                 # Naming
                 base_name = "unknown_function",
                 name=None,
                 output_file = None,
                 # Specification
                 io_precision = ML_Binary32, 
                 abs_accuracy = None,
                 libm_compliant = True,
                 # Optimization parameters
                 processor = GenericProcessor(),
                 fuse_fma = True, 
                 fast_path_extract = True,
                 # Debug verbosity
                 debug_flag = False
             ):
        
        self.io_precision = io_precision
        # Naming logic, using provided information if available, otherwise deriving from base_name
        # base_name is e.g. exp
        # name is e.g. expf or expd or whatever 
        self.base_name = base_name
        if name:
            self.name = name
        else:
            newname = self.base_name
            if self.io_precision==ML_Binary32:
                newname +="f"
            self.name = newname
            if output_file:
                self.output_file = output_file
            else:
                self.output_file = self.name + ".c"
                
        self.debug_flag=debug_flag

        self.sollya_precision = self.io_precision.sollya_object

        if abs_accuracy:
            self.abs_accuracy = abs_accuracy
        else:
            self.abs_accuracy = S2**(-self.io_precision.get_precision())

        self.libm_compliant = libm_compliant
        
        self.processor = processor

        self.fuse_fma = fuse_fma
        self.fast_path_extract = fast_path_extract

        self.implementation = CodeFunction(self.base_name, output_format = self.io_precision)
        self.opt_engine = OptimizationEngine(self.processor)




    def generate_C(self):
        """Final C generation, once the evaluation scheme has been optimized"""
        # registering scheme as function implementation
        self.implementation.set_scheme(self.evalScheme)
        self.C_code_generator = CCodeGenerator(self.processor, declare_cst = False, disable_debug = not self.debug_flag, libm_compliant = self.libm_compliant)
        self.result = self.implementation.get_definition(self.C_code_generator, C_Code, static_cst = True)
        #self.result.add_header("support_lib/ml_special_values.h")
        self.result.add_header("math.h")
        self.result.add_header("stdio.h")
        self.result.add_header("inttypes.h")
        #print self.result.get(self.C_code_generator)

        Log.report(Log.Info, "Generating C code in " + self.implementation.get_name() + ".c")
        output_stream = open("%s.c" % self.implementation.get_name(), "w")
        output_stream.write(self.result.get(self.C_code_generator))
        output_stream.close()


    # Currently mostly empty, to be populated someday
    def gen_emulation_code(self, precode, code, postcode):
        """generate C code that emulates the function, typically using MPFR.
        precode is declaration code (before the test loop)
        postcode is clean-up code (after the test loop)
        Takes the input and output names from input_list and output_list.
        Must postfix output names with "ref_", "ref_ru_", "ref_rd_"


        This class method performs commonly used initializations. 
        It initializes the MPFR versions of the inputs and outputs, 
        with the same names prefixed with "mp" and possibly postfixed with "rd" and "ru".

        It should be overloaded by actual metafunctions, and called by the overloading function. 
        """


        
