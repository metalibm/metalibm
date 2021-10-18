

=== Static Vectorizer


=== VLAVectorizer

The VLA vectorizer targets Vector Length Agnostic (VLA) architectures such as ARM's VLA or RISC-V V extensions.
The generate code is agnostic of the actual datapath or vector register lengths.

VLA support introduces the following new operations:
- `VLABlock(length, statement) -> ML_Void` statement is executed assuming a vector size of `length`
- `VLAGetLength(reqLength) -> ML_Integer` request `reqLength` vector size and returns the actual vector supported

VLA support introduces the following formats:
- `VLAType(<base_type>)` vector of agnostic length with `base_type` as element type. The actual vector length is defined by the operation environnement.

VLA support introduces the following construct:
- `VLAOp(<op>, <vl>)`

No operation expecting `VLAType` parameters or returning `VLAType`  result can be defined outside a `VLABlock`.