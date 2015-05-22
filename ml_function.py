from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.code_element import CodeFunction
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.ml_optimization_engine import OptimizationEngine



class ML_Function:
    """A class from which all metafunction inherit"""
    # basename is e.g. exp
    # name is e.g. expf or expd or whatever 
    basename = "unknown_function"
    def __init__(self,
                 basename = "unknown_function",
                 precision = ML_Binary32,
                 processor = GenericProcessor() 
             ):
        self.precision = precision
        sollya_precision = self.precision.sollya_object
        self.implementation = CodeFunction(self.basename, output_format = self.precision)
        self.processor = processor
        self.opt_engine = OptimizationEngine(self.processor)


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

