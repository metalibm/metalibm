
import asmde.allocator as asmde

from metalibm_core.core.ml_operations import Return, is_leaf_node, Constant, TableStore
from metalibm_core.core.legalizer import is_constant
from metalibm_core.core.bb_operations import (
    BasicBlockList, BasicBlock,
    UnconditionalBranch, ConditionalBranch)
from metalibm_core.core.machine_operations import (
    MachineRegister, RegisterAssign, PhysicalRegister)


from metalibm_core.utility.log_report import Log

class BackendAsmArchitecture(asmde.Architecture):
    """ bridge Metalibm with asmde.Architecture class """
    pass


def extract_src_regs_from_node(node):
    """ extract the list of MachineRegister objects used by node """
    reg_list = [reg for reg in node.inputs if isinstance(reg, MachineRegister)]
    assert all(map(lambda r: isinstance(r, (MachineRegister, Constant)), reg_list))
    return reg_list


class AssemblySynthesizer:
    def __init__(self, arch):
        # map of metalibm register to their asmde counterparts
        # each metalibm register is associated to a tuple of asmde registers
        self.ml_to_asmde_reg_map = {}
        self.ml_to_physical_reg_map = {}
        self.architecture = arch

    def generate_allocatable_register(self, ml_reg):
        """ """
        assert isinstance(ml_reg, MachineRegister)
        if ml_reg in self.ml_to_asmde_reg_map:
            return self.ml_to_asmde_reg_map[ml_reg]
        else:
            # TODO/FIXME: manage multiple register class
            # TODO/FIXME: manage physical register
            virt_reg = self.architecture.generate_virtual_reg(ml_reg)
            self.ml_to_asmde_reg_map[ml_reg] = virt_reg
            return virt_reg

    def get_physical_reg(self, color_map, ml_reg, reg_class=asmde.Register.Std):
        if not ml_reg in self.ml_to_physical_reg_map:
            # TODO/FIXME: currently only support a single-reg tuple
            asmde_regs = self.ml_to_asmde_reg_map[ml_reg]
            reg_ids = [color_map[reg_class][asmde_reg] for asmde_reg in asmde_regs]
            physical_reg = PhysicalRegister(reg_ids, ml_reg.precision, ml_reg.get_tag(),
                                            var_tag=ml_reg.var_tag)
            self.ml_to_physical_reg_map[ml_reg] = physical_reg
        return self.ml_to_physical_reg_map[ml_reg]

    def transform_to_physical_reg(self, color_map, linearized_program):
        """ transform each MachineRegister in linearized_program into
            the corresponding PhysicalRegister """
        for bb in linearized_program.inputs:
            for node in bb.inputs:
                if isinstance(node, RegisterAssign):
                    # dst reg
                    node.set_input(0, self.get_physical_reg(color_map, node.get_input(0)))
                    # src regs
                    value_node = node.get_input(1)
                    if not is_leaf_node(value_node):
                        for index, op in enumerate(value_node.inputs):
                            if isinstance(op, MachineRegister):
                                value_node.set_input(index, self.get_physical_reg(color_map, op))
                            elif is_constant(op):
                                value_node.set_input(index, op)
                            else:
                                raise NotImplementedError
                    elif isinstance(value_node, MachineRegister):
                        node.set_input(1, self.get_physical_reg(color_map, value_node))
                    elif is_constant(value_node):
                        node.set_input(1, value_node)
                    else:
                        raise NotImplementedError
                elif isinstance(node, ConditionalBranch):
                    op = node.get_input(0)
                    if isinstance(op, MachineRegister):
                        node.set_input(0, self.get_physical_reg(color_map, op))
                    else:
                        raise NotImplementedError
                elif isinstance(node, TableStore):
                    for index, op in enumerate(node.inputs):
                        if isinstance(op, MachineRegister):
                            node.set_input(index, self.get_physical_reg(color_map, op))
                        elif is_constant(op):
                            node.set_input(index, op)
                        else:
                            raise NotImplementedError


    def generate_insn_from_node(self, node):
        """ generate a asmde.Instruction which corresponds to node """
        if isinstance(node, RegisterAssign):
            dst_reg = node.get_input(0)
            dst_reg_list = [sub_reg for reg in [dst_reg] for sub_reg in self.generate_allocatable_register(reg)]
            value_node = node.get_input(1)
            if is_constant(value_node):
                src_reg_list = []
            else:
                src_reg_list = [sub_reg for reg in extract_src_regs_from_node(value_node) for sub_reg in self.generate_allocatable_register(reg) ]
            insn = asmde.Instruction(node,
                               dbg_object=node,
                               def_list=dst_reg_list,
                               use_list=src_reg_list)
            return insn

        elif isinstance(node, TableStore):
            # TODO/FIXME: may need the generation of a shaddow dependency chain
            # to maintain TableStore/TableLoad relative order when required
            src_reg_list = [sub_reg for reg in extract_src_regs_from_node(node) for sub_reg in self.generate_allocatable_register(reg) ]
            insn = asmde.Instruction(node,
                                dbg_object=node,
                               use_list=src_reg_list)
            return insn

        else:
            Log.report(Log.Error, "node unsupported in AssemblySynthesizer.generate_insn_from_node: {}", node)
            raise NotImplementedError

    def generate_ABI_phys_input_regs(self, ordered_input_regs):
        """ generate a list of physical register to store Program inputs
            matching ABI constraints """
        phys_tuple_list = self.architecture.generate_ABI_physical_input_reg_tuples(ordered_input_regs)
        for ml_reg, asmde_reg_tuple in zip(ordered_input_regs, phys_tuple_list):
            self.ml_to_asmde_reg_map[ml_reg] = asmde_reg_tuple
        phys_reg_list = [reg for reg_tuple in phys_tuple_list for reg in reg_tuple]
        return phys_reg_list

    def generate_ABI_phys_output_regs(self, ordered_output_regs):
        """ generate a list of physical register to store Program outputs
            matching ABI constraints """
        phys_tuple_list = self.architecture.generate_ABI_physical_output_reg_tuples(ordered_output_regs)
        for ml_reg, asmde_reg_tuple in zip(ordered_output_regs, phys_tuple_list):
            self.ml_to_asmde_reg_map[ml_reg] = asmde_reg_tuple
        return [reg for reg_tuple in phys_tuple_list for reg in reg_tuple]

    def translate_to_asmde_program(self, linearized_program, input_reg_list, output_reg_list):
        """ Translate a linearized program (BasicBlockList)
            into a asmde.Program object """


        #pre_defined_list = [self.generate_allocatable_register(reg) for reg in input_reg_list]
        #post_used_list = [self.generate_allocatable_register(reg) for reg in output_reg_list]

        # implement ABI
        pre_defined_list = self.generate_ABI_phys_input_regs(input_reg_list)
        post_used_list = self.generate_ABI_phys_output_regs(output_reg_list)
        program = asmde.Program(pre_defined_list=pre_defined_list,
                                post_used_list=post_used_list, empty=True)

        # mapping of Metalibm's BasicBlock to asmde's ones
        ml_bb_to_asmde_bb = {}
        class UnresolvedLink:
            """ structure to store a Metalibm's BasicBlock which has not yet
                an ASMDE counterpart. The structure also store ASMDE the
                predecessors of this block. Link between the ASDME counterpart
                and predecessors must be created once the ASMDE counterpart
                can be resolved """
            def __init__(self, ml_bb, predecessors=None):
                self.ml_bb = ml_bb
                self.predecessors = predecessors

        # register unresolved-links (Metalibm's BasicBlock which
        # could not be resolved to asmde BasicBlock, because of forward
        # declaration)
        list_unresolved_links = []

        assert isinstance(linearized_program, BasicBlockList)
        for bb in linearized_program.inputs:
            asmde_bb = program.add_new_current_bb(bb.get_tag())
            ml_bb_to_asmde_bb[bb] = asmde_bb
            for node in bb.inputs:
                # TODO/FIXME: for now one bundle per instruction
                bundle = asmde.Bundle()
                if isinstance(node, Return):
                    assert len(node.inputs) == 0
                    # return does not use any register
                    insn = asmde.Instruction(node, dbg_object=node)
                    # return means the current basic block is a parent of
                    # the sink (exit)
                    program.current_bb.add_successor(program.sink_bb)
                    program.sink_bb.add_predecessor(program.current_bb)

                elif isinstance(node, (UnconditionalBranch, ConditionalBranch)):
                    for ml_bb_succ in node.destination_list:
                        if not ml_bb_succ in ml_bb_to_asmde_bb:
                            list_unresolved_links.append(
                                UnresolvedLink(ml_bb_succ, [program.current_bb])
                            )

                        else:
                            bb_succ = ml_bb_to_asmde_bb[ml_bb_succ]
                            #program.current_bb.add_successor(bb_succ)
                            #bb_succ.add_predecessor(program.current_bb)
                            program.current_bb.connect_to(bb_succ)
                    use_list = []
                    if isinstance(node, ConditionalBranch):
                        # ConditionalBranch use a register
                        cond = node.get_input(0)
                        # NOTES: transform tuple into list 
                        use_list = list(self.generate_allocatable_register(cond))

                    insn = asmde.Instruction(node, use_list=use_list, is_jump=True, dbg_object=node)

                else:
                    insn = self.generate_insn_from_node(node)
                bundle.add_insn(insn)
                program.add_bundle(bundle)

        # resolving links
        for unresolved_link in list_unresolved_links:
            ml_bb = unresolved_link.ml_bb
            try:
                asmde_bb = ml_bb_to_asmde_bb[ml_bb]
            except KeyError:
                print(ml_bb_to_asmde_bb)
                Log.report(Log.Error, "could not find asmde counterpart for Metalibm's BB {}", ml_bb)
            for predecessor in unresolved_link.predecessors:
                #predecessor.add_successor(asmde_bb)
                #asmde_bb.add_predecessor(predecessor)
                predecessor.connect_to(asmde_bb)

        # connecting first BasicBlock to source BasicBlock
        program.source_bb.connect_to(ml_bb_to_asmde_bb[linearized_program.inputs[0]])

        program.end_program()
        return program

    def perform_register_allocation(self, program):
        print("Register Assignation")
        reg_assignator = asmde.RegisterAssignator(self.architecture)

        empty_liverange_map = self.architecture.get_empty_liverange_map()

        var_ins, var_out = reg_assignator.generate_use_def_lists(program)
        liverange_map = reg_assignator.generate_liverange_map(program, empty_liverange_map, var_ins, var_out)

        print("Checking liveranges")
        liverange_status = reg_assignator.check_liverange_map(liverange_map)

        print("Graph coloring")
        conflict_map = reg_assignator.create_conflict_map(liverange_map)
        color_map = reg_assignator.create_color_map(conflict_map)
        for reg_class in conflict_map:
            conflict_graph = conflict_map[reg_class]
            class_color_map = color_map[reg_class]
            check_status = reg_assignator.check_color_map(conflict_graph, class_color_map)
            if not check_status:
                Log.report(Log.Error, "register assignation for class {} does is not valid", reg_class)

        return color_map

