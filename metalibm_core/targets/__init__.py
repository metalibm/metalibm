# -*- coding: utf-8 -*-
import os


def target_validity_test(target_name):
    return os.path.isdir(target_name)

# dynamically search for installed targets
targets_dirname = os.path.dirname(os.path.realpath(__file__))

target_list = [possible_target for possible_target in os.listdir(targets_dirname) if target_validity_test(os.path.join(targets_dirname, possible_target))]
    
__all__ = target_list

# listing submodule

if __name__ == "__main__":
    print "target_list: ", target_list
