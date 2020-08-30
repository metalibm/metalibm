from metalibm_core.core.ml_operations import (
    Variable, ReferenceAssign, ML_ArithmeticOperation,
)

from metalibm_core.core.bb_operations import BasicBlockList, BasicBlock


"""

specification for machine-instruction program

The program is a MachineProgram inhereting from BasicBlockList

each BasicBlock must be a list of RegisterAssign or PhiNode

each RegisterAssign must assign the results of an operation between MachineRegister-s to
a MachineRegister. the operation depth must be less than or equals to 1 (copy are allowed).
That means each operand of an operation must be either a MachineRegister of a leaf node (Constant).

"""

class MachineProgram(BasicBlockList):
    """ Machine program """
    @property
    def ordered_input_regs(self):
        return self._ordered_input_regs
    @ordered_input_regs.setter
    def ordered_input_regs(self, ordered_input_regs):
        self._ordered_input_regs = ordered_input_regs

    def finish_copy(self, new_copy, copy_map=None):
        """ Propagating final attribute during copy """
        new_copy.ordered_input_regs = self.ordered_input_regs

class MachineRegister(Variable):
    """ Machine register """
    name = "MachineRegister"
    physical = False
    def __init__(self, register_id, register_format, reg_tag, var_tag=None, **kw):
        """ register tag is stored as inner Variable's name
            and original variable's name is stored in self.var_tag """
        reg_tag = "unamed-reg" if reg_tag is None else reg_tag
        Variable.__init__(self, reg_tag, precision=register_format, **kw)
        self.var_tag = var_tag
        self.register_id = register_id

class SubRegister(MachineRegister):
    """ sub-chunk of a machine register """
    def __init__(self, super_register, sub_id, register_id, register_format, reg_tag, **kw):
        MachineRegister.__init__(self, register_id, register_format, reg_tag, super_register.var_tag, **kw)
        self.super_register = super_register
        self.sub_id = sub_id

class PhysicalRegister(MachineRegister):
    name = "PhysicalRegister"
    physical = True

class RegisterAssign(ReferenceAssign):
    name = "RegisterAssign"
    arity = 2

class RegisterCopy(ML_ArithmeticOperation):
    """ Copy a register into a virtual register value
        (to be used in a RegisterAssign for a physical copy) """
    name = "RegisterCopy"
    arity = 1

class MaterializeConstant(ML_ArithmeticOperation):
    """ Explicitly materialize a constant value into a register """
    name = "MaterializeConstant"
    arity = 1

class MaterializeConstant(ReferenceAssign):
    name = "MaterializeConstant"

