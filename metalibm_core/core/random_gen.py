# -*- coding: utf-8 -*-
""" metalibm_core.core.random_gen Random value generation """

from enum import Enum, unique
import random

from sollya import SollyaObject, S2

from metalibm_core.core.ml_formats import (
    FP_PlusInfty, FP_MinusInfty, FP_PlusZero, FP_MinusZero,
    FP_PlusOmega, FP_MinusOmega, FP_QNaN, FP_SNaN
)

import metalibm_core.core.ml_formats as ml_formats


def normalize_map(weight_map):
    """ Ensure that every weight in map is positive and adds up to 1.0.
        Works inplace
    """
    total = 0.0
    # summing
    for key in weight_map:
        weight = weight_map[key]
        assert weight >= 0
        total += weight
    # normalizing
    normalization_factor = float(total)
    assert normalization_factor > 0
    for key in weight_map:
        weight_map[key] = weight_map[key] / normalization_factor
    return weight_map


class FPRandomGen(object):
    """ Random generator for floating-point numbers """
    @unique # pylint: disable=too-few-public-methods
    class Category(Enum):
        """ Value set category """
        ##  Special value category
        SpecialValues = 0
        ## Subnormal numbers category
        Subnormal = 1
        ## Normal numbers category
        Normal = 2

    special_value_ctor = [
        FP_PlusInfty, FP_MinusInfty,
        FP_PlusZero, FP_MinusZero,
        FP_PlusOmega, FP_MinusOmega,
        FP_QNaN, FP_SNaN
    ]
    def __init__(self, precision, weight_map=None, seed=None):
        """
            Args:
                precision (ML_Format): floating-point format
                weight_map (dict): map category -> weigth

        """
        self.precision = precision
        self.weight_map = normalize_map({
            FPRandomGen.Category.SpecialValues: 0.1,
            FPRandomGen.Category.Subnormal: 0.2,
            FPRandomGen.Category.Normal: 0.7,

        } if weight_map is None else weight_map)
        self.category_keys = self.weight_map.keys()
        self.random = random.Random(seed)
        self.sp_list = self.get_special_value_list()


    def get_special_value_list(self):
        """ Returns a list a special values in the generator precision """
        return  [
            sp_class(self.precision) for sp_class in
            FPRandomGen.special_value_ctor
        ]

    def generate_special_value(self):
        """ Generate a single special value """
        sp_index = self.random.randrange(len(self.sp_list))
        return self.sp_list[sp_index]

    def generate_sign(self):
        """ Generate a random sign value {-1.0, 1.0} """
        return SollyaObject(-1.0) if self.random.randrange(2) == 1 else \
               SollyaObject(1.0)

    def generate_normal_number(self):
        """ Generate a single value in the normal range """
        field_size = self.precision.get_field_size()
        exp = self.random.randrange(
            self.precision.get_emin_normal(),
            self.precision.get_emax() + 1
        )
        sign = self.generate_sign()
        field = self.random.randrange(2**field_size)
        mantissa = 1.0 + field * S2**-self.precision.get_field_size()
        return mantissa * sign * S2**exp

    def generate_subnormal_number(self):
        """ Generate a single subnormal value """
        field_size = self.precision.get_field_size()
        # a subnormal has the same exponent as the minimal normal
        # but without implicit 1.0 digit
        exp = self.precision.get_emin_normal()
        sign = self.generate_sign()
        field = self.random.randrange(2**field_size)
        mantissa = 0.0 + field * S2**-self.precision.get_field_size()
        return mantissa * sign * S2**exp

    def get_new_value_by_category(self, category):
        """ generate a new value from the given category """
        gen_map = {
            FPRandomGen.Category.SpecialValues: self.generate_special_value,
            FPRandomGen.Category.Normal: self.generate_normal_number,
            FPRandomGen.Category.Subnormal: self.generate_subnormal_number
        }
        gen_func = gen_map[category]
        return gen_func()

    def get_category_from_weight_index(self, weight_index):
        """ returns the set category corresponding to weight_index """
        for category in self.category_keys:
            weight_index -= self.weight_map[category]
            if weight_index <= 0.0:
                return category
        return self.category_keys[0]

    def get_new_value(self):
        """ Generate a new random value """
        weight_index = self.random.random()
        category = self.get_category_from_weight_index(weight_index)
        return self.get_new_value_by_category(category)


# auto-test
if __name__ == "__main__":
    RG = FPRandomGen(ml_formats.ML_Binary32)
    for i in xrange(20):
        value = RG.get_new_value()
        print value
    RG = FPRandomGen(ml_formats.ML_Binary64)
    for i in xrange(20):
        value = RG.get_new_value()
        print value
