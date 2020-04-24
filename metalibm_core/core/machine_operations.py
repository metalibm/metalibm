from metalibm_core.core.ml_operations import (
    Variable, ReferenceAssign
)


"""

specification for machine-instruction program

The program is a BasicBlockList

each BasicBlockList must be a list of RegisterAssign or PhiNode

each RegisterAssign must assign the results of an operation between MachineRegister-s to
a MachineRegister. the operation depth must be less than or equals to 1 (copy are allowed).
That means each operand of an operation must be either a MachineRegister of a leaf node (Constant).

"""

class MachineRegister(Variable):
    """ Machine register """
    name = "MachineRegister"
    def __init__(self, register_id, register_format, reg_tag, var_tag=None, **kw):
        """ register tag is stored as inner Variable's name
            and original variable's name is stored in self.var_tag """
        Variable.__init__(self, reg_tag, precision=register_format, **kw)
        self.var_tag = var_tag

class RegisterAssign(ReferenceAssign):
    name = "RegisterAssign"

class MaterializeConstant(ReferenceAssign):
    name = "MaterializeConstant"
