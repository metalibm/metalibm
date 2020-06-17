MDL provides a certain number of utilities to describe fixed-point operation.

### The fixed-point formats

The HDL fixed-point format is accessible throught the `fixed_point` helper function implemented in the `core.ml_hdl_formats` module.

### The FixedPointPosition node
The `FixedPointPosition` is a class of nodes available when describing a meta-entity. This node does not correspond to an operation of the implementation graph but is an index (whose actual value is resolved during the code generation) within a fixed-point format.
This constructor can be used to determine the position of the 'point' in a datum of unknown width (unknown at the time of the meta entity description).

The FixedPoint operation is part of the `core.advanced_operations` module. It should be instanciated as follows.
```
FixedPoint(node, value, align)
```
Where `node` is an operation node, value is an `offset` (positive or negative integer) and `align` is an alignment specificer.
The node value is resolved with respect to `node`, `value` and `align` as follows:
- `align=`**FixedPointPosition.FromMSBToLSB**: the result is the index of the digit at lower position `value` with respect to `node`'s MSB  and computed from the LSB position.
- `align=`**FixedPointPosition.FromLSBToLSB**: The result is the index of the digit at position `value` with respect to `node`'s  LSB and computed from the LSB position which means the result is equals to value.
- `align=`**FixedPointPosition.FromPointToLSB**: The result is computed as the index of the digit at higher position `value` with respect to `node`'s zero point (digit with weight 0) and computed from the LSB position.
- `align=`**FixedPointPosition.FromPointToMSB**: The result is computed as the index of the digit at position `value` with respect to `node`'s zero point (digit with weight 0) and computed from the MBS position.

### FixedPointPosiition evaluation

Beware that FixedPointPosition's value is only evaluated during code generation, thus during meta-entity entity description it will not be evaluated as a constant.
The evaluation is performed by the **rtl_legalize** optimization pass; together with the **size_datapath** optimization pass they can be used to legalize fixed-point operation graphs.

### Example 

Let us present several examples. In all those examples we indicate what will be the value of the FixedPointPosition during code generation. As described in the previous section, the FixedPointPosition node will not hold this value at its instanciation (during the meta entity description).

```python
# 15-bit unsigned fixed-point format with 5-bit fractionnal part and 
#  10-bit integer part
fp_format = fixed_point(10, 5, signed=False)

# let us assume node A is of precision fp_format

point_index = FixedPointPosition(A, 0, align=FixedPointPosition.FromPointToLSB)
# we compute the position of the fixed-point (position=0) with respect to the datum LSB position
>>> 5

point_index = FixedPointPosition(A, 3, align=FixedPointPosition.FromPointToLSB)
# we compute the position of the digit of weigth 2^3 (position=3) with respect to the datum LSB position
>>> 8

point_index = FixedPointPosition(A, 0, align=FixedPointPosition.FromMSBToLSB)
# we compute the position of the most significant digit with respect to the datum LSB position
>>> 15

point_index = FixedPointPosition(A, 17, align=FixedPointPosition.FromLSBToLSB)
# we compute the position of the digit of weight = 2^17*(weight(LSB)) with respect to the datum LSB position
>>> 17
```
