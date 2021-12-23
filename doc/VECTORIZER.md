

=== Static Vectorizer


=== VLAVectorizer

The VLA vectorizer targets Vector Length Agnostic (VLA) architectures such as ARM's SVE/SVE2 or RISC-V V extensions.
The generate code is agnostic of the actual datapath or vector register lengths.

VLA support introduces the following new operations:
- `VLAGetLength(reqLength) -> ML_Integer` request `reqLength` vector size and returns the actual vector supported
- `VLAOperation(*ops, vl, specifier=<>, **kw)` this operation wraps a standard operation, adding an extra vector length parameter `vl`. The wrapped operation is specified by passing its class as specifier argument.

VLA support introduces the following formats:
- `VLAType(<base_type>)` vector of agnostic length with `base_type` as element type. The actual vector length is defined by the operation environnement.
