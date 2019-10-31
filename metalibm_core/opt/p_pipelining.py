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
""" Optimization pass to generate pipelined meta entity """
import re

from metalibm_core.utility.log_report import Log

from metalibm_core.core.ml_formats import ML_Bool
from metalibm_core.core.ml_operations import (
    LogicalAnd, ConditionBlock, Comparison, Statement, ML_LeafNode,
    Variable, ReferenceAssign, Constant
)
from metalibm_core.core.ml_hdl_operations import (
    Process, Event, Signal, StaticDelay
)
from metalibm_core.core.advanced_operations import FixedPointPosition
from metalibm_core.core.ml_hdl_format import ML_StdLogic


class RetimeMap:
    """This is class to store retimed signal map (correspondance between signal
       and stage) """
    def __init__(self):
        # map (op_key, stage) -> stage's op
        self.stage_map = {}
        # map of stage_index -> list of pipelined forward
        # from <stage_index> -> <stage_index + 1>
        self.stage_forward = {}
        # list of nodes already retimed
        self.processed = set()
        #
        self.pre_statement = set()
        # number of flip/flops (bit registers)
        self.register_count = 0

    def get_op_key(self, op):
        """ get the RetimeMap's key associated with op

            :param op: operand
            :type op: metalibm_core.core.ML_Operation """
        op_key = op.attributes.init_op if not op.attributes.init_op is None else op
        return op_key

    def hasBeenProcessed(self, op):
        """ test if op has already been processed

            :param op: input node
            :type op: ML_Operation
            :return: true if node has been processed, false otherwise
            :rtype: bool """
        return self.get_op_key(op) in self.processed

    def addToProcessed(self, op):
        """ add op to the list of processed nodes

            :param op: input node
            :type op: ML_Operation
        """
        op_key = self.get_op_key(op)
        return self.processed.add(op_key)

    def contains(self, op, stage):
        """ check if the pair (op, stage) is defined in the stage map

            :param op: input node
            :type op: ML_Operation
            :param stage: input pipeline stage index
            :type stage: int
            :return: true if (op, stage) is defined in stage map
            :rtype: bool """
        return (self.get_op_key(op), stage) in self.stage_map

    def get(self, op, stage):
        """ get the delayed node associated to the key pair (op, stage) in the stage map

            :param op: input node
            :type op: ML_Operation
            :param stage: input pipeline stage index
            :type stage: int
            :return: delayed node
            :rtype: ML_Operation """
        return self.stage_map[(self.get_op_key(op), stage)]

    def set(self, op, stage):
        """ define the delayed node in stage map

            :param op: delayed version of node
            :type op: ML_Operation
            :param stage: input pipeline stage index
            :type stage: int
        """
        op_key = self.get_op_key(op)
        self.stage_map[(op_key, stage)] = op

    def add_stage_forward(self, op_dst, op_src, stage):
        """ Adding a node to forward op_src to op_dst at stage index

            :param op_dst: stage output node (register output)
            :type op_dst: ML_Operation
            :param op_src: stage input node (register entry)
            :type op_src: ML_Operation
            :param stage: destination stage index
            :type stage: int
        """
        Log.report(Log.Verbose, " adding stage forward {op_src} to {op_dst} @ stage {stage}",
            op_src=op_src, op_dst=op_dst, stage=stage)
        if not stage in self.stage_forward:
            self.stage_forward[stage] = []
        self.stage_forward[stage].append(
            ReferenceAssign(op_dst, op_src)
        )
        self.register_count += op_dst.get_precision().get_bit_size()
        self.pre_statement.add(op_src)

def propagate_op(op, stage, retime_map):
    """ propagate the node op until stage by forwarding it through the pipeline
        and adding intermediate register at each stage crossed

        :param op: node to be propagated in the pipeline
        :type op: ML_Operation
        :param stage: index of the final pipeline stage
        :type stage: int
        :param retime_map: retiming-map to use to store propagation steps
        :type retime_map: RetimeMap
    """
    op_key = retime_map.get_op_key(op)
    Log.report(Log.Verbose, " propagating {op} (key={op_key}) to stage {stage}",
        op=op, op_key=op_key, stage=stage)
    # look for the latest stage where op is defined
    current_stage = op_key.attributes.init_stage
    while retime_map.contains(op_key, current_stage + 1):
        current_stage += 1
    op_src = retime_map.get(op_key, current_stage)
    while current_stage != stage:
        # create op instance for <current_stage+1>
        # remove cycle information prefix on tag if any
        raw_tag = re.sub("^e[0-9]+_", "", op_key.get_tag() or "empty")
        op_dst = Signal(tag="e{stage}_{tag}_q".format(
            tag=raw_tag, stage=(current_stage + 2)),
            init_stage=current_stage + 1, init_op=op_key,
            precision=op_key.get_precision(), var_type=Variable.Local)
        retime_map.add_stage_forward(op_dst, op_src, current_stage)
        retime_map.set(op_dst, current_stage + 1)
        # update values for next iteration
        current_stage += 1
        op_src = op_dst


def node_should_be_pipelined(optree):
    """ Predicate to determined if a node should be pipelined when forwarded
        accross stages or can be propagated as is (e.g. Constant should not
        be propagated)

        :param optree: input node
        :type optree: ML_Operation
        :return: true if node should be pipelined, false otherwise
        :rtype: bool
    """
    if isinstance(optree, FixedPointPosition):
        return False
    elif isinstance(optree, Constant):
        return False
    else:
        return True


def node_has_inputs(optree):
    """ Predicate checking if node has inputs

        :param optree: input node
        :type optree: ML_Operation
        :return: true if optree is not a leaf node
        :rtype: bool
    """
    if isinstance(optree, ML_LeafNode):
        return False
    else:
        return True


def retime_op(op, retime_map):
    """ Process each input of op and if necessary generate necessary
        forwarding stage

        :param op: input node
        :type op: ML_Operation
        :param retime_map: retiming map
        :type retime_map: RetimeMap
    """
    op_stage = op.attributes.init_stage
    Log.report(Log.Verbose, "retiming op [S={}] {} ", op_stage, op)
    if retime_map.hasBeenProcessed(op):
        Log.report(Log.Verbose, "  retiming already processed")
        return

    if isinstance(op, StaticDelay):
        pass
    elif node_has_inputs(op):
        for in_id in range(op.get_input_num()):
            in_op = op.get_input(in_id)
            in_stage = in_op.attributes.init_stage
            Log.report(
                Log.Verbose,
                "retiming input {inp} of {op} stage {in_stage} -> {op_stage}",
                inp=in_op, op=op, in_stage=in_stage,
                op_stage=op_stage
            )
            if not retime_map.hasBeenProcessed(in_op):
                retime_op(in_op, retime_map)

            if not node_should_be_pipelined(in_op):
                pass

            elif isinstance(in_op, StaticDelay):
                # Squashing delta delays
                delay_op = in_op.get_input(0)
                delay_value = in_op.delay
                if not retime_map.hasBeenProcessed(delay_op):
                    retime_op(delay_op, retime_map)
                if in_op.relative:
                    op_stage = delay_op.attributes.init_stage + delay_value
                else:
                    op_stage = delay_value
                Log.report(Log.Verbose, "squashing StaticDelay on {delay_op}, delay={delay}",
                    delay_op=delay_op, delay=delay_value
                )
                if not retime_map.contains(delay_op, op_stage):
                    propagate_op(delay_op, op_stage, retime_map)
                new_in = retime_map.get(delay_op, op_stage)
                op.set_input(in_id, new_in)
            elif in_stage < op_stage:
                assert not isinstance(in_op, FixedPointPosition)
                if not retime_map.contains(in_op, op_stage):
                    propagate_op(in_op, op_stage, retime_map)
                new_in = retime_map.get(in_op, op_stage)
                Log.report(Log.Verbose, "new version of input {inp} for {op} is {new_in}",
                    inp=in_op, op=op, new_in=new_in)
                op.set_input(in_id, new_in)
            elif in_stage > op_stage:
                Log.report(Log.Error, "stages {in_stage} -> {op_stage}, input {inp} of {op} is defined at a later stage".format(
                    in_stage=in_stage, op_stage=op_stage,
                    inp=in_op.get_str(
                        display_precision=True,
                        custom_callback=lambda op: " [S={}] ".format(
                            op.attributes.init_stage)
                    ),
                    op=op.get_str(
                        display_precision=True,
                        custom_callback=lambda op: " [S={}] ".format(
                            op.attributes.init_stage)
                    )
                )
                )
    retime_map.set(op, op_stage)
    retime_map.addToProcessed(op)


def generate_pipeline_stage(entity, reset=False, recirculate=False, one_process_per_stage=True, synchronous_reset=True, negate_reset=False):
    """ Process a entity to generate pipeline stages required to implement
        pipeline structure described by node's stage attributes.

        :param entity: input entity to pipeline
        :type entity: ML_EntityBasis
        :param reset: indicate if a reset must be generated for pipeline registers
        :type reset: bool
        :param recirculate: trigger the integration of a recirculation signal to the stage
            flopping condition
        :type recirculate: bool
        :param one_process_per_stage:forces the generation of a separate process for each
               pipeline stage (else a unique process is generated for all the stages
        :type one_process_per_stage: bool
        :param synchronous_reset: triggers the generation of a clocked reset
        :type synchronous_reset: bool
        :param negate_reset: if set indicates the reset is triggered when reset signal is 0
                            (else 1)
        :type negate_reset: bool
    """
    retiming_map = {}
    retime_map = RetimeMap()
    output_assign_list = entity.implementation.get_output_assign()
    for output in output_assign_list:
        Log.report(Log.Verbose, "generating pipeline from output {} ", output)
        retime_op(output, retime_map)
    for recirculate_stage in entity.recirculate_signal_map:
        recirculate_ctrl = entity.recirculate_signal_map[recirculate_stage]
        Log.report(Log.Verbose, "generating pipeline from recirculation control signal {}", recirculate_ctrl)
        retime_op(recirculate_ctrl, retime_map)

    process_statement = Statement()

    # adding stage forward process
    clk = entity.get_clk_input()
    clock_statement = Statement()
    global_reset_statement = Statement()


    Log.report(Log.Info, "design has {} flip-flop(s).", retime_map.register_count)

    # handle towards the first clock Process (in generation order)
    # which must be the one whose pre_statement is filled with
    # signal required to be generated outside the processes
    first_process = False
    for stage_id in sorted(retime_map.stage_forward.keys()):
        stage_statement = Statement(
            *tuple(assign for assign in retime_map.stage_forward[stage_id]))

        if reset:
            reset_statement = Statement()
            for assign in retime_map.stage_forward[stage_id]:
                target = assign.get_input(0)
                reset_value = Constant(0, precision=target.get_precision())
                reset_statement.push(ReferenceAssign(target, reset_value))

            if recirculate:
                # inserting recirculation condition
                recirculate_signal = entity.get_recirculate_signal(stage_id)
                stage_statement = ConditionBlock(
                    Comparison(
                        recirculate_signal,
                        Constant(0, precision=recirculate_signal.get_precision()),
                        specifier=Comparison.Equal,
                        precision=ML_Bool
                    ),
                    stage_statement
                )

            if synchronous_reset:
                # build a compound statement with reset and flops statement
                stage_statement = ConditionBlock(
                    Comparison(
                        entity.reset_signal,
                        Constant(0 if negate_reset else 1, precision=ML_StdLogic),
                        specifier=Comparison.Equal, precision=ML_Bool
                    ),
                    reset_statement,
                    stage_statement
                )
            else:
                # for asynchronous reset, reset is in a non-clocked statement
                # and will be added at the end of stage to the same process than
                # register clocking
                global_reset_statement.add(reset_statement)

        # To meet simulation / synthesis tools, we build
        # a single if clock predicate block per stage
        clock_block = ConditionBlock(
            LogicalAnd(
                Event(clk, precision=ML_Bool),
                Comparison(
                    clk,
                    Constant(1, precision=ML_StdLogic),
                    specifier=Comparison.Equal,
                    precision=ML_Bool
                ),
                precision=ML_Bool
            ),
            stage_statement
        )

        if one_process_per_stage:
            if reset and not synchronous_reset:
                clock_block = ConditionBlock(
                    Comparison(
                        entity.reset_signal,
                        Constant(0 if negate_reset else 1, precision=ML_StdLogic),
                        specifier=Comparison.Equal, precision=ML_Bool
                    ),
                    reset_statement,
                    clock_block
                )
                clock_process = Process(clock_block, sensibility_list=[clk, entity.reset_signal])

            else:
                # no reset, or synchronous reset (already appended to clock_block)
                clock_process = Process(clock_block, sensibility_list=[clk])
            entity.implementation.add_process(clock_process)

            first_process = first_process or clock_process
        else:
            clock_statement.add(clock_block)
    if one_process_per_stage:
        # reset and clock processed where generated at each stage loop
        pass
    else:
        process_statement.add(clock_statement)
        if synchronous_reset:
            pipeline_process = Process(process_statement, sensibility_list=[clk])
        else:
            process_statement.add(global_reset_statement)
            pipeline_process = Process(process_statement, sensibility_list=[clk, entity.reset_signal])
        entity.implementation.add_process(pipeline_process)
        first_process = pipeline_process
    # statement that gather signals which must be pre-computed
    for op in retime_map.pre_statement:
        first_process.add_to_pre_statement(op)
    stage_num = len(retime_map.stage_forward.keys())
    Log.report(Log.Info, "there are {} pipeline stage(s)", stage_num)
    return stage_num
