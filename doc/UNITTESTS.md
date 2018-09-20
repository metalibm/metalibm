

## Metalibm Unitary testing ##

### Executing unit-test ###

```
python valid/soft_unit_test.py
python valid/rtl_unit_test.py
```

### Adding new unit tests ###

For software related features: **metalibm_functions/unit_tests/**
For hardware related features: **metalibm_hw_blocks/unit_tests/**



### Defining a new test ###

The MetaFunction containing the test should be declared as a class with **TestRunner** as a parent.
TestRunner declares the specific API to generate a unit-test. 
It is also import that the MetaFunction implements the **get_default_args** static method (as examplified below)
which gives TestRunner the possibility to determine a meaningful value for all parameters.
The TestRunner child shall also implement a **__call__** static method which will be called to execute the test.

```python
from metalibm_functions.unit_tests.utils import TestRunner
from metalibm_functions.core.ml_function import ML_FunctionBasis

class NewUnitTest(ML_FunctionBasis, TestRunner):
    function_name = "NewUnitTest"
	def __init__(self, args):
    	....
        
  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "libm_compliant": True
    }
    default_args.update(kw)
    return DefaultArgTemplate(**default_args)

   # execution function for the TestRunner class
   def __call__(args):
        meta_instance = NewUnitTest(args)
        return True
        
 run_test = NewUnitTest

```

### Registerting a new test ###

Software test should be added in **valid/soft_unit_test.py** script by extending the **unit_test_list** with a new entry:

```python
import metalibm_functions.unit_tests.new_unit_test as ut_new_unit_test

...

unit_test_list = [
  ... other tests ...
  # new test 
  UnitTestScheme(
    # succint test description
    "new unit test",
    # test modules
    ut_new_unit_test,
    # list of test argument, each dictionnary will generate a separate test
    # and the dictionnary content will be used to overload default arguments
    [{"passes": ["beforecodegen:gen_basic_block", "beforecodegen:dump", "beforecodegen:ssa_translation"]}]
  ),
]
```
