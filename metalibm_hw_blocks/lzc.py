# -*- coding: utf-8 -*-

import sys

import sollya

from sollya import S2, Interval, ceil, floor, round, inf, sup, log, exp, expm1, log2, guessdegree, dirtyinfnorm, RN, RD
from sollya import parse as sollya_parse

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_Table
import metalibm_core.code_generation.vhdl_backend as vhdl_backend
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_entity import ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed

from metalibm_core.utility.rtl_debug_utils import debug_dec


from metalibm_core.core.ml_hdl_format import *
from metalibm_core.core.ml_hdl_operations import *

class ML_LeadingZeroCounter(ML_Entity("ml_lzc")):
  @staticmethod
  def get_default_args(width = 32):
    return DefaultEntityArgTemplate( 
             precision = ML_Int32, 
             debug_flag = False, 
             target = vhdl_backend.VHDLBackend(), 
             output_file = "my_lzc.vhd", 
             entity_name = "my_lzc",
             language = VHDL_Code,
             width = width,
           )

  def __init__(self, arg_template = None):
    # building default arg_template if necessary
    arg_template = ML_LeadingZeroCounter.get_default_args() if arg_template is None else arg_template
    # initializing I/O precision
    self.width = arg_template.width
    precision = arg_template.precision
    io_precisions = [precision] * 2
    Log.report(Log.Info, "generating LZC with width={}".format(self.width))

    # initializing base class
    ML_EntityBasis.__init__(self, 
      base_name = "ml_lzc",
      arg_template = arg_template
    )

    self.accuracy  = arg_template.accuracy
    self.precision = arg_template.precision

  def numeric_emulate(self, io_map):
    def count_leading_zero(v, w):
      tmp = v
      lzc = -1
      for i in range(w):
        if tmp & 2**(w - 1 - i):
          return i
      return w
    result = {}
    result["vr_out"] = count_leading_zero(io_map["x"], self.width)
    return result

  def generate_scheme(self):
    lzc_width = int(floor(log2(self.width))) + 1
    Log.report(Log.Info, "width of lzc out is {}".format(lzc_width))
    input_precision = ML_StdLogicVectorFormat(self.width)
    precision = ML_StdLogicVectorFormat(lzc_width)
    # declaring main input variable
    vx = self.implementation.add_input_signal("x", input_precision) 
    vr_out = Signal("lzc", precision = precision, var_type = Variable.Local)
    tmp_lzc = Variable("tmp_lzc", precision = precision, var_type = Variable.Local)
    iterator = Variable("i", precision = ML_Integer, var_type = Variable.Local)
    lzc_loop = RangeLoop(
      iterator,
      Interval(0, self.width - 1),
      ConditionBlock(
        Comparison(
          VectorElementSelection(vx, iterator, precision = ML_StdLogic),
          Constant(1, precision = ML_StdLogic),
          specifier = Comparison.Equal,
          precision = ML_Bool
        ),
        ReferenceAssign(
          tmp_lzc,
          Conversion(
            Subtraction(
              Constant(self.width - 1, precision = ML_Integer),
              iterator,
              precision = ML_Integer
            ),
          precision = precision),
        )
      ),
      specifier = RangeLoop.Increasing,
    )
    lzc_process = Process(
      Statement(
        ReferenceAssign(tmp_lzc, Constant(self.width, precision = precision)),
        lzc_loop,
        ReferenceAssign(vr_out, tmp_lzc)
      ),
      sensibility_list = [vx]
    )

    self.implementation.add_process(lzc_process)

    self.implementation.add_output_signal("vr_out", vr_out)

    return [self.implementation]

  standard_test_cases =[sollya_parse(x) for x in  ["1.1", "1.5"]]

def vhdl_legalize_count_leading_zeros(optree):
    """ Legalize a CountLeadingZeros node into a valid vhdl 
        implementation

        Args:
            optree (CountLeadingZeros): input node

        Return:
            ML_Operation: legal operation graph to implement LZC
    """
    lzc_format = optree.get_precision()
    lzc_input = optree.get_input(0)
    lzc_width = lzc_input.get_precision().get_bit_size()

    lzc_args = ML_LeadingZeroCounter.get_default_args(width = lzc_width)
    LZC_entity = ML_LeadingZeroCounter(lzc_args)
    lzc_entity_list = LZC_entity.generate_scheme()
    lzc_implementation = LZC_entity.get_implementation()

    lzc_component = lzc_implementation.get_component_object()

    lzc_tag = optree.get_tag() if not optree.get_tag() is None else "lzc_signal"

    # LZC output value signal
    lzc_signal = Signal(
        lzc_tag, precision = lzc_format,
        var_type = Signal.Local, debug = debug_dec
    )
    lzc_value = PlaceHolder(
        lzc_signal,
        lzc_component(io_map = {
            "x": lzc_input, 
            "vr_out": lzc_signal
        }, tag = "lzc_i"), tag = "place_holder"
    )
    # returing PlaceHolder as valid leading zero count result
    return lzc_value


Log.report(Log.Info, "installing ML_LeadingZeroCounter legalizer in vhdl backend")
vhdl_backend.handle_LZC_legalizer.optree_modifier = vhdl_legalize_count_leading_zeros

if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(default_entity_name = "new_lzc", default_output_file = "ml_lzc.vhd", default_arg = ML_LeadingZeroCounter.get_default_args())
    arg_template.parser.add_argument("--width", dest = "width", type=int, default = 32, help = "set input width value (in bits)")
    # argument extraction 
    args = parse_arg_index_list = arg_template.arg_extraction()

    ml_lzc           = ML_LeadingZeroCounter(args)

    ml_lzc.gen_implementation()
