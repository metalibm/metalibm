

Metalibm distinguishes precision and accuracy

Precision is an operation format, meaning the size and encoding of its inputs and output.

Accuracy is a measure of an operation error relative to a perfect (i.e. exact) version of the same operation.

## Possible accuracies

Accuracies can be defined in terms of error (absolute or relative).

In floating-point, an import metric for the relative error is the ulp (unit in last place), which roughly corresponds to the weight of the least significant bit of the number's mantissa.

For a more serious definition of the ulp and its many variants, please refer to: 
- [On the definition of ulp(x), Jean-Michel Muller](http://www.ens-lyon.fr/LIP/Pub/Rapports/RR/RR2005/RR2005-09.pdf)

### Correctly Rounded

Less than half a "ulp"

On metalibm command line, such accuracy can be indicated by:
    --accuracy cr

WARNING: Currently no meta-function supports the generation of a correctly rounded version.
The `cr` option can still be used to test any implementation (or even an external function) against a correctlt-rounded version and highlight differences.

### Faithfully rounded

Less than an ulp
On metalibm command line, such accuracy can be indicated by:
    --accuracy faithful

### Degraded accuracy

This type of accuracy is used to indicate anything that does not fit into any of the previous categories

On metalibm command line, such accuracy can be indicated by:
    --accuracy 'daa(goal)' # degraded accuracy absolute with abs(value - exact) < goal
    --accuracy 'dar(goal)' # degraded accuracy relative with abs ((value - exact) / exact) < goal
