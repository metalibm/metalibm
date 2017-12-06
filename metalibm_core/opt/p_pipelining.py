from metalibm_core.utility.log_report import Log

from metalibm_core.core.ml_formats import ML_Bool
from metalibm_core.core.ml_operations import (
    LogicalAnd, ConditionBlock, Comparison, Statement, ML_LeafNode,
    Variable, ReferenceAssign, Constant
)
from metalibm_core.core.ml_hdl_operations import (
    Process, Event, Signal
)
from metalibm_core.core.advanced_operations import FixedPointPosition
from metalibm_core.core.ml_hdl_format import ML_StdLogic


class RetimeMap:
    def __init__(self):
        # map (op_key, stage) -> stage's op
        self.stage_map = {}
        # map of stage_index -> list of pipelined forward
        # from <stage_index> -> <stage_index + 1>
        self.stage_forward = {}
        # list of nodes already retimed
        self.processed = []
        #
        self.pre_statement = set()

    def get_op_key(self, op):
        op_key = op.attributes.init_op if not op.attributes.init_op is None else op
        return op_key

    def hasBeenProcessed(self, op):
        """ test if op has already been processed """
        return self.get_op_key(op) in self.processed

    def addToProcessed(self, op):
        """ add op to the list of processed nodes """
        op_key = self.get_op_key(op)
        return self.processed.append(op_key)

    def contains(self, op, stage):
        """ check if the pair (op, stage) is defined in the stage map """
        return (self.get_op_key(op), stage) in self.stage_map

    def get(self, op, stage):
        return self.stage_map[(self.get_op_key(op), stage)]

    def set(self, op, stage):
        op_key = self.get_op_key(op)
        self.stage_map[(op_key, stage)] = op

    def add_stage_forward(self, op_dst, op_src, stage):
        Log.report(Log.Verbose, " adding stage forward {op_src} to {op_dst} @ stage {stage}".format(
            op_src=op_src, op_dst=op_dst, stage=stage))
        if not stage in self.stage_forward:
            self.stage_forward[stage] = []
        self.stage_forward[stage].append(
            ReferenceAssign(op_dst, op_src)
        )
        self.pre_statement.add(op_src)

# propagate forward @p op until it is defined
#  in @p stage
def propagate_op(op, stage, retime_map):
    op_key = retime_map.get_op_key(op)
    Log.report(Log.Verbose, " propagating {op} (key={op_key}) to stage {stage}".format(
        op=op, op_key=op_key, stage=stage))
    # look for the latest stage where op is defined
    current_stage = op_key.attributes.init_stage
    while retime_map.contains(op_key, current_stage + 1):
        current_stage += 1
    op_src = retime_map.get(op_key, current_stage)
    while current_stage != stage:
        # create op instance for <current_stage+1>
        op_dst = Signal(tag="{tag}_S{stage}".format(
            tag=op_key.get_tag(), stage=(current_stage + 1)),
            init_stage=current_stage + 1, init_op=op_key,
            precision=op_key.get_precision(), var_type=Variable.Local)
        retime_map.add_stage_forward(op_dst, op_src, current_stage)
        retime_map.set(op_dst, current_stage + 1)
        # update values for next iteration
        current_stage += 1
        op_src = op_dst


def node_should_be_pipelined(optree):
    """ Predicate to authorize pipeline to node """
    if isinstance(optree, FixedPointPosition):
        return False
    elif isinstance(optree, Constant):
        return False
    else:
        return True


def node_has_inputs(optree):
    if isinstance(optree, ML_LeafNode): 
        return False
    else:
        return True


def retime_op(op, retime_map):
    """ Process each input of op and if necessary generate necessary
        forwarding stage """
    Log.report(Log.Verbose, "retiming op %s " % (op.get_str(depth=1)))
    if retime_map.hasBeenProcessed(op):
        Log.report(Log.Verbose, "  retiming already processed")
        return
    op_stage = op.attributes.init_stage
    if node_has_inputs(op):
        for in_id in range(op.get_input_num()):
            in_op = op.get_input(in_id)
            in_stage = in_op.attributes.init_stage
            Log.report(
                Log.Verbose,
                "retiming input {inp} of {op} stage {in_stage} -> {op_stage}".format(
                    inp=in_op.get_str(depth=1), op=op, in_stage=in_stage,
                    op_stage=op_stage
                )
            )
            if not retime_map.hasBeenProcessed(in_op):
                retime_op(in_op, retime_map)

            if not node_should_be_pipelined(in_op):
                pass
            elif in_stage < op_stage:
                assert not isinstance(in_op, FixedPointPosition)
                if not retime_map.contains(in_op, op_stage):
                    propagate_op(in_op, op_stage, retime_map)
                new_in = retime_map.get(in_op, op_stage)
                Log.report(Log.Verbose, "new version of input {inp} for {op} is {new_in}".format(
                    inp=in_op, op=op, new_in=new_in))
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


def generate_pipeline_stage(entity):
    """ Process a entity to generate pipeline stages required """
    retiming_map = {}
    retime_map = RetimeMap()
    output_assign_list = entity.implementation.get_output_assign()
    for output in output_assign_list:
        Log.report(Log.Verbose, "generating pipeline from output %s " %
                   (output.get_str(depth=1)))
        retime_op(output, retime_map)
    process_statement = Statement()

    # adding stage forward process
    clk = entity.get_clk_input()
    clock_statement = Statement()
    for stage_id in sorted(retime_map.stage_forward.keys()):
            stage_statement = Statement(
                *tuple(assign for assign in retime_map.stage_forward[stage_id]))
            clock_statement.add(stage_statement)
    # To meet simulation / synthesis tools, we build
    # a single if clock predicate block which contains all
    # the stage register allocation
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
        clock_statement
    )
    process_statement.add(clock_block)
    pipeline_process = Process(process_statement, sensibility_list=[clk])
    for op in retime_map.pre_statement:
        pipeline_process.add_to_pre_statement(op)
    entity.implementation.add_process(pipeline_process)
    stage_num = len(retime_map.stage_forward.keys())
    #print "there are %d pipeline stages" % (stage_num)
    return stage_num 
