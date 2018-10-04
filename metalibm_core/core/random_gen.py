# -*- coding: utf-8 -*-
""" metalibm_core.core.random_gen Random value generation """

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2018 Kalray
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

try:
    from enum import Enum, unique
except ImportError:
    class Enum(object): pass

    def unique(fct):
        """ Fallback for unique decorator when unique is not defined """
        return fct
import random

from sollya import SollyaObject
S2 = SollyaObject(2)

from metalibm_core.core.special_values import (
    NumericValue,
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


class RandomGenWeightCat(object):
    """ Abstract random number generator using weighted
        categories """
    def __init__(self, category_keys=None, weight_map=None):
        self.category_keys = category_keys
        self.weight_map = weight_map

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
        return category.generate_value(self)


class IntRandomGen(RandomGenWeightCat):
    """ Random generator for integer values (signed and unsigned) """
    class Category:
        """ Integer value categories """
        class MaxValue:
            """ generate the maximal format value """
            @staticmethod
            def generate_value(generator):
                return generator.gen_max_value()
        class MinValue:
            """ generate the minimal format value """
            @staticmethod
            def generate_value(generator):
                return generator.gen_min_value()
        class ZeroValue:
            """ generate a zero value """
            @staticmethod
            def generate_value(generator):
                return generator.gen_zero_value()
        class HighValue:
            """ generate a value close to MaxValue """
            @staticmethod
            def generate_value(generator):
                return generator.gen_high_value()
        class LowValue:
            """ generate a value close to MinValue """
            @staticmethod
            def generate_value(generator):
                return generator.gen_low_value()
        class NearZero:
            @staticmethod
            def generate_value(generator):
                """ generate near zero value """
                return generator.gen_near_zero()
        class Standard:
            @staticmethod
            def generate_value(generator):
                """ generate value arbitrarily in the whole format range """
                return generator.gen_standard_value()
    def __init__(self, size=32, signed=True, seed=None):
        """ Initializing Integer random generators """
        int_weight_map = normalize_map({
            IntRandomGen.Category.MaxValue: 0.01,
            IntRandomGen.Category.MinValue: 0.01,
            IntRandomGen.Category.ZeroValue: 0.01,
            IntRandomGen.Category.HighValue: 0.07,
            IntRandomGen.Category.LowValue: 0.07,
            IntRandomGen.Category.NearZero: 0.07,
            IntRandomGen.Category.Standard: 0.76,
        })
        category_keys = int_weight_map.keys()
        RandomGenWeightCat.__init__(
            self,
            category_keys=category_keys,
            weight_map=int_weight_map,
        )
        self.signed = signed
        self.size = size
        self.random = random.Random(seed)
        # subrange for fuzzing value around specific values
        self.highlow_range = 2**(int(self.size / 5))

    def gen_max_value(self):
        """ generate the maximal format value """
        power = self.size - (1 if self.signed else 0)
        return S2**power - 1
    def gen_min_value(self):
        """ generate the minimal format value """
        if self.signed:
            return -S2**(self.size - 1)
        else:
            return self.gen_zero_value()
    def gen_zero_value(self):
        """ generate zero value """
        return 0
    def gen_high_value(self):
        """ generate near maximal value """
        return self.gen_max_value() - self.random.randrange(self.highlow_range)
    def gen_low_value(self):
        """ generate new minimal value """
        return self.gen_min_value() + self.random.randrange(self.highlow_range)

    def gen_near_zero(self):
        """ generate near zero value """
        if self.signed:
            start_value = self.gen_zero_value() - self.highlow_range
            random_offset = self.random.randrange(self.highlow_range * 2)
            return start_value + random_offset
        else:
            return self.gen_low_value()
    def gen_standard_value(self):
        """ generate value arbitrarily in the whole format range """
        return self.random.randrange(self.gen_min_value(), self.gen_max_value() + 1)


class FixedPointRandomGen(IntRandomGen):
    def __init__(self, int_size=1, frac_size=31, signed=True, seed=None):
        """ Initializing Integer random generators """
        int_weight_map = normalize_map({
            IntRandomGen.Category.MaxValue: 0.01,
            IntRandomGen.Category.MinValue: 0.01,
            IntRandomGen.Category.ZeroValue: 0.01,
            IntRandomGen.Category.HighValue: 0.07,
            IntRandomGen.Category.LowValue: 0.07,
            IntRandomGen.Category.NearZero: 0.07,
            IntRandomGen.Category.Standard: 0.76,
        })
        category_keys = int_weight_map.keys()
        RandomGenWeightCat.__init__(
            self,
            category_keys=category_keys,
            weight_map=int_weight_map, 
        )
        self.signed = signed
        self.int_size = int_size
        self.frac_size = frac_size
        self.size = int_size + frac_size
        self.random = random.Random(seed)
        # subrange for fuzzing value around specific values
        self.highlow_range = 2**(int(self.size / 5))

    def scale(self, value):
        return value * S2**-self.frac_size

    def gen_max_value(self, scale=True):
        """ generate the maximal format value """
        scale_func = (lambda x: self.scale(x)) if scale else (lambda x: x)
        power = self.size - (1 if self.signed else 0)
        return scale_func(S2**power - 1)
    def gen_min_value(self, scale=True):
        """ generate the minimal format value """
        scale_func = (lambda x: self.scale(x)) if scale else (lambda x: x)
        if self.signed:
            return scale_func(-S2**(self.size - 1))
        else:
            return scale_func(self.gen_zero_value())
    def gen_zero_value(self):
        """ generate zero value """
        return 0
    def gen_high_value(self):
        """ generate near maximal value """
        return self.scale(self.gen_max_value(scale=False) - self.random.randrange(self.highlow_range))
    def gen_low_value(self):
        """ generate new minimal value """
        return self.scale(self.gen_min_value(scale=False) + self.random.randrange(self.highlow_range))

    def gen_near_zero(self):
        """ generate near zero value """
        if self.signed:
            start_value = self.gen_zero_value() - self.highlow_range
            random_offset = self.random.randrange(self.highlow_range * 2)
            return self.scale(start_value + random_offset)
        else:
            return self.scale(self.gen_low_value())
    def gen_standard_value(self):
        """ generate value arbitrarily in the whole format range """
        return self.scale(
            self.random.randrange(
                self.gen_min_value(scale=False),
                self.gen_max_value(scale=False) + 1
            )
        )


class FPRandomGen(RandomGenWeightCat):
    """ Random generator for floating-point numbers """
    class Category:
        """ Value set category """
        ##  Special value category
        class SpecialValues:
            """ Special values """
            @staticmethod
            def generate_value(generator):
                """ Generate a single special value """
                return random.choice(generator.sp_list)

        class Subnormal:
            """ Subnormal numbers """
            @staticmethod
            def generate_value(generator):
                """ Generate a single subnormal value """
                field_size = generator.precision.get_field_size()
                # a subnormal has the same exponent as the minimal normal
                # but without implicit 1.0 digit
                exp = generator.precision.get_emin_normal()
                sign = generator.generate_sign()
                field = generator.random.randrange(2**field_size)
                mantissa = 0.0 + field * S2**-generator.precision.get_field_size()
                return NumericValue(mantissa * sign * S2**exp)

        class Normal:
            """ Normal number category """
            @staticmethod
            def generate_value(generator):
                """ Generate a single value in the normal range """
                field_size = generator.precision.get_field_size()
                exp = generator.random.randrange(
                    generator.precision.get_emin_normal(),
                    generator.precision.get_emax() + 1
                )
                sign = generator.generate_sign()
                field = generator.random.randrange(2**field_size)
                mantissa = 1.0 + field * S2**-generator.precision.get_field_size()
                return NumericValue(mantissa * sign * S2**exp)

    special_value_ctor = [
        FP_PlusInfty, FP_MinusInfty,
        FP_PlusZero, FP_MinusZero,
        FP_PlusOmega, FP_MinusOmega,
        FP_QNaN, FP_SNaN
    ]
    def __init__(self, precision, weight_map=None, seed=None, generation_map=None):
        """
            Args:
                precision (ML_Format): floating-point format
                weight_map (dict): map category -> weigth

        """
        self.precision = precision

        weight_map = normalize_map({
            FPRandomGen.Category.SpecialValues: 0.1,
            FPRandomGen.Category.Subnormal: 0.2,
            FPRandomGen.Category.Normal: 0.7,

        } if weight_map is None else weight_map)
        category_keys = weight_map.keys()
        RandomGenWeightCat.__init__(
            self,
            weight_map=weight_map,
            category_keys = category_keys
        )

        self.random = random.Random(seed)
        self.sp_list = self.get_special_value_list()


    def get_special_value_list(self):
        """ Returns a list a special values in the generator precision """
        return  [
            sp_class(self.precision) for sp_class in
            FPRandomGen.special_value_ctor
        ]


    def generate_sign(self):
        """ Generate a random sign value {-1.0, 1.0} """
        return SollyaObject(-1.0) if self.random.randrange(2) == 1 else \
               SollyaObject(1.0)



# auto-test
if __name__ == "__main__":
    RG = FPRandomGen(ml_formats.ML_Binary32)
    for i in range(20):
        value = RG.get_new_value()
        print(value)
    RG = FPRandomGen(ml_formats.ML_Binary64)
    for i in range(20):
        value = RG.get_new_value()
        print(value)

    IG = IntRandomGen(37, signed=True)
    for i in range(20):
        value = IG.get_new_value()
        print(value)

    FIXG = FixedPointRandomGen(int_size=3, frac_size=15, signed=True)
    for i in range(20):
        value = FIXG.get_new_value()
        print(value)

