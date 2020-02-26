# Metalibm Description Language

Metalibm Description Language (MDL) is the entry point for Metalibm.
An implementation is built using MDL and static generation allowed by the Metalibm framework (e.g. generating table content using pythonsollya). The power of metalibm lies on mixing mathematical and code generation tools to generate an efficient implementation.

# Basics MDL constructs
MDL is based on mix of control flow and dataflow nodes.
Dataflow takes the form of operation expressions using nodes such as **Addition**, **Multiplication**, **Variable**, **Constant**.
Control flow uses nodes such as **ConditionBlock**, **SwitchBlock** or **Return**.

### Expression
MDL expressions are a composition of leaf nodes (**Constant**, **Variable**) and operation nodes (**Addition**, **Multiplication**, ...).

    # a variable called x with undefined precision
    x = Variable("x")
    # Addition between x and the numerical value 3.
    # Metalibm will promote 3 from numerical value to Constant node
    y = Addition(x, 3)
    # The previous expression can be written with implicit operations
    z = x + 3


### Statement
Control flow can be described in MDL using Statement constructs such as **ConditionBlock**, **Statement**, **Return**.

    # if (cond) then return a else return b
    scheme = ConditionBlock(
	    cond,
	    Return(a),
	    Return(b)
	)

### Variable and scope

MDL provides several variable construct.
By default a variable is an input to the current scheme, it is considered alive everywhere (declared outside the scope of the scheme).
A Variable node can be declared of **var_type** Variable.Local, it such a case its scope is reduced.
If a variable first appears in an expression is scope is limited to the Statement surrounding the expression.
If a variable first appears alone in a Statement is scope is the statement and every encapsulated sub-statement.

### Tables


## Attributes

### Precision
MDL Nodes can be annotated with attributes which specifies their properties. For example you can specify the precision of a node by using **precision** attribute.

    # a single precision variable called "t"
    t = Variable("t", precision=ML_Binary32)

Basic MDL precisions include: **ML_Binary32** (float), **ML_Binary64** (double), **ML_Int32** (int32_t), **ML_UInt32** (uint32_t*), **ML_UInt64**, **ML_Int64**.
MDL also contains compound precisions: **ML_DoubleDouble**, **ML_TripleDouble** and vector formats: **v\<i\>float32** (vector of single precision elements),  **v<i>float64** (vector of double precision elements), **v\<i\>[u]int32**, **v\<i\>[u]int64**, **v\<i\>bool** with **\<i\>** in 2, 3, 4, 8. 

### Tag
You can force the name of a node by using the **tag** attribute.

    # An addition tagged "unique_sum", this name should appear in the generated code
    unique_sum = Addition(t, x, tag="unique_sum")

