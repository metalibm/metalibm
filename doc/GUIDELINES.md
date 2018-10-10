# Guidelines to developer


## Message logging


### Warning
Metalibm uses its own message logging system.
No **print** should appear in any module or meta-function of metalibm.

### Log module

Logging is provided through ** metalibm_core.utility.log_report ** module
through the Log object.

    # importing logging module
    from metalibm_core.utility.log_report import Log

    # reporting a message with Information Level
    Log.report(Log.Info, "my informative message")

    # reporting an error (by default an exception is also raised)
    Log.report(Log.Error, "my error message")


On command-line, log levels can be enabled through the verbose options

    --verbose Info,Verbose,Warning
