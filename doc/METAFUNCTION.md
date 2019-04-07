
Metalibm can be extended by adding more meta-function description.
Example of meta-functions can be found under the metalibm_functions/ directory.


## New meta-function ##

A Meta-function should be declared as a new Class inheriting from **ML_FunctionBasis** class
The static class property **function_name** should contain the function name.

```
from metalibm_core.core.ml_function import ML_FunctionBasis

class NewMetaFunction(ML_FunctionBasis):
	function_name = "new_meta_function"
```

### Meta-function description ###

The meta-function implementation must be described using the Meta Description Language (see doc/MDL.md).
The easiest way is to overload the **generate_scheme** method. This method take a single argument (self) and returns the main expression of the meta-function (most often a Statement node).

For example if we want our new meta-function to describe the addition of two parameters

```
def generate_scheme(self):
	# declare a new input parameters vx whose tag is "x" and whose precision is single precision
	vx = self.implementation.add_input_variable("x", ML_Binary32)
    
    # declare a new input parameters vy whose tag is "y" and whose precision is single precision
	vy = self.implementation.add_input_variable("x", ML_Binary32)
    
    main_scheme = Statement(
    	Return(vx + vy, precision=ML_Binary32)
    )
    return main_scheme
```

### Parametrisable meta-function ###

Among the many standard parameters which can be given through command-line two are of uttermost interest:
- **--input-precision** defines the functions parameters precisions
- **--precision** defines the function result precision

In the meta-function class the output precision can be accessed through **self.precision** and input precision and be accessed through **self.get_input_precision(param_index)** where **param_index** is the index of the parameter one wishes to access the precision of.

We can modify our scheme generation to use those paramaters.
```
def generate_scheme(self):
	# declare a new input parameters vx whose tag is "x" and whose precision is single precision
    vx = self.implementation.add_input_variable("x", self.get_input_precision(0))
    
    # declare a new input parameters vy whose tag is "y" and whose precision is single precision
	vy = self.implementation.add_input_variable("x", self.get_input_precision(1))
    
    # convert parameters to make sure addition format are uniform
    vx = Conversion(vx, precision=self.precision)
    vy = Conversion(vy, precision=self.precision)
    
    main_scheme = Statement(
    	Return(vx + vy, precision=self.precision)
    )
    return main_scheme
```


### Function command line arguments ###

Standard arguments from the commande line (see  doc/USERGUIDE.md) can be used with the new meta_functions.

```
if __name__ == "__main__":
    # declaring standard argument structure
    arg_template = ML_NewArgTemplate()

    # filling arg_template structure with command line options
    args = arg_template.arg_extraction()

	# declaring meta-function instance
    meta_function = NewMetaFunction(args)

	# generating meta_function
    meta_function.gen_implementation()
```

### Describing meta-function numerical model ###

To indicate to metalibm engine how to emulate the function behavior (e.g. to generate expected test outputs), you must overload the **numeric_emulate** method. This method takes as many arguments as the meta-function.

For example for our new meta-function with two parameters:
```
	# class NewMetaFunction
	def numerical_emulate(self, x, y):
    	return x + y
```

By default the numerical model must be in arbitrary precision, metalibm will build test and expected value based on function's precision and accuracy parameter.


### Complete code example ###

```
from metalibm_core.core.ml_function import ML_FunctionBasis
from metalibm_core.core.ml_formats import ML_Binary32

class NewMetaFunction(ML_FunctionBasis):
	function_name = "new_meta_function"

	def generate_scheme(self):
        # declare a new input parameters vx whose tag is "x" and whose precision is single precision
        vx = self.implementation.add_input_variable("x", self.get_input_precision(0))

    	# declare a new input parameters vy whose tag is "y" and whose precision is single precision
        vy = self.implementation.add_input_variable("x", self.get_input_precision(0))

    	main_scheme = Statement(
    		Return(vx + vy, precision=ML_Binary32)
    	)
    	return main_scheme


if __name__ == "__main__":
    # declaring standard argument structure
    arg_template = ML_NewArgTemplate()

    # filling arg_template structure with command line options
    args = arg_template.arg_extraction()

	# declaring meta-function instance
    meta_function = NewMetaFunction(args)

	# generating meta_function
    meta_function.gen_implementation()
```

```
	# class NewMetaFunction
	def numerical_emulate(self, x, y):
    	return x + y
```
