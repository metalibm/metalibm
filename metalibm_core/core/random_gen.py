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

import sollya
S2 = sollya.SollyaObject(2)

from metalibm_core.core.special_values import (
    NumericValue,
    FP_PlusInfty, FP_MinusInfty, FP_PlusZero, FP_MinusZero,
    FP_PlusOmega, FP_MinusOmega, FP_QNaN, FP_SNaN
)


import metalibm_core.core.ml_formats as ml_formats
from metalibm_core.core.ml_formats import (
    ML_FP_MultiElementFormat, ML_FP_Format, ML_Fixed_Format,
)

from metalibm_core.utility.log_report import Log

class RandomDescriptor:
    """ descriptor for a random generator descriptor """
    pass

class UniformInterval(RandomDescriptor):
    """ descriptor for uniform random generation """
    def __init__(self, lo, hi):
        self.interval = sollya.Interval(lo, hi)

    def __repr__(self):
        return "U[{};{}]".format(sollya.inf(self.interval), sollya.sup(self.interval))

def random_bool():
    """ random boolean generation """
    return bool(random.getrandbits(1))


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
        return self.rectify_value(self.gen_max_value() - self.random.randrange(self.highlow_range))
    def gen_low_value(self):
        """ generate new minimal value """
        return self.rectify_value(self.gen_min_value() + self.random.randrange(self.highlow_range))
    def rectify_value(self, value):
        """ Transform value so that it fits within valid range """
        return max(self.min_value, min(self.max_value, value))

    def gen_near_zero(self):
        """ generate near zero value """
        if self.signed:
            start_value = self.gen_zero_value() - self.highlow_range
            random_offset = self.random.randrange(self.highlow_range * 2)
            return self.rectify_value(start_value + random_offset)
        else:
            return self.gen_low_value()
    def gen_standard_value(self):
        """ generate value arbitrarily in the whole format range """
        return self.random.randrange(self.gen_min_value(), self.gen_max_value() + 1)


class FixedPointRandomGen(IntRandomGen):
    def __init__(self, int_size=1, frac_size=31, signed=True, seed=None, min_value=None, max_value=None):
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

        # scaled extremal values
        self.max_value = self.gen_raw_max_value() if max_value is None else max_value
        self.min_value = self.gen_raw_min_value() if min_value is None else min_value
        # unscaled extremal values (used for random field generation)
        self.max_unscaled_value = self.unscale(self.max_value)
        self.min_unscaled_value = self.unscale(self.min_value)

    def scale(self, value):
        return value * S2**-self.frac_size
    def unscale(self, value):
        return value * S2**self.frac_size

    def gen_raw_max_value(self, scale=True):
        """ generate the maximal format value """
        scale_func = (lambda x: self.scale(x)) if scale else (lambda x: x)
        power = self.size - (1 if self.signed else 0)
        return scale_func(S2**power - 1)
    def gen_raw_min_value(self, scale=True):
        """ generate the minimal format value """
        scale_func = (lambda x: self.scale(x)) if scale else (lambda x: x)
        if self.signed:
            return scale_func(-S2**(self.size - 1))
        else:
            return scale_func(self.gen_zero_value())
    def gen_min_value(self, scale=True):
        return self.rectify_value(self.gen_raw_min_value(scale))
    def gen_max_value(self, scale=True):
        return self.rectify_value(self.gen_raw_max_value(scale))
    def gen_zero_value(self):
        """ generate zero value """
        return 0
    def gen_high_value(self):
        """ generate near maximal value """
        return self.rectify_value(self.max_value - self.scale(self.random.randrange(self.highlow_range)))
    def gen_low_value(self):
        """ generate new minimal value """
        return self.rectify_value(self.min_value + self.scale(self.random.randrange(self.highlow_range)))

    def gen_near_zero(self):
        """ generate near zero value """
        if self.signed:
            start_value = self.gen_zero_value() - self.highlow_range
            random_offset = self.random.randrange(self.highlow_range * 2)
            return self.rectify_value(self.scale(start_value + random_offset))
        else:
            return self.scale(self.gen_low_value())
    def gen_standard_value(self):
        """ generate value arbitrarily in the whole format range """
        return self.scale(
            self.random.randrange(
                self.min_unscaled_value,
                self.max_unscaled_value + 1
            )
        )
    @staticmethod
    def from_interval(precision, low_bound, high_bound):
        return FixedPointRandomGen(
            precision.integer_size,
            precision.frac_size,
            precision.signed,
            min_value=low_bound, max_value=high_bound
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
        class ZeroExp:
            """ number with zero valued exponent """
            @staticmethod
            def generate_value(generator):
                """ Generate a single value in the normal range """
                field_size = generator.precision.get_field_size()
                field = generator.random.randrange(2**field_size)
                mantissa = 1.0 + field * S2**-generator.precision.get_field_size()
                return NumericValue(mantissa)
        class FPLogInterval:
            """ interval number category """
            def __init__(self, inf_bound, sup_bound):
                self.inf_bound = inf_bound
                self.sup_bound = sup_bound
                self.zero_in_interval = 0 in sollya.Interval(inf_bound, sup_bound)
                self.min_exp = None if self.zero_in_interval else min(sollya.ceil(sollya.log2(abs(inf_bound))), sollya.ceil(sollya.log2(abs(sup_bound))))
                self.max_exp = max(sollya.ceil(sollya.log2(abs(inf_bound))), sollya.ceil(sollya.log2(abs(sup_bound))))

            def generate_value(self, generator):
                # TODO/FIXME random.uniform only generate a machine precision
                # number (generally a double) which may not be suitable
                # for larger format
                field_size = generator.precision.get_field_size()
                exp = generator.random.randrange(
                    generator.precision.get_emin_normal() if self.min_exp is None else self.min_exp,
                    (generator.precision.get_emax() + 1) if self.max_exp is None else (self.max_exp + 1)
                )
                sign = generator.generate_sign()
                field = generator.random.randrange(2**field_size)
                mantissa = 1.0 + field * S2**-generator.precision.get_field_size()
                random_value = mantissa * sign * S2**exp
                return NumericValue(min(max(self.inf_bound, random_value), self.sup_bound))

        class FPUniformInterval(FPLogInterval):
            """ interval number category with uniform generation """

            def generate_value(self, generator):
                # TODO/FIXME random.uniform only generate a machine precision
                # number (generally a double) which may not be suitable
                # for larger format
                # Nonetheless this number must be rounded(-down) to the generator
                # precision to avoid double-rounding issue down the line
                return NumericValue(generator.precision.round_sollya_object(random.uniform(self.inf_bound, self.sup_bound)))


                # value = generator.precision.round_sollya_object(random.uniform(self.inf_bound, self.sup_bound))
                # return NumericValue(value)

    special_value_ctor = [
        FP_PlusInfty, FP_MinusInfty,
        FP_PlusZero, FP_MinusZero,
        FP_PlusOmega, FP_MinusOmega,
        FP_QNaN, FP_SNaN
    ]
    def __init__(self, precision, weight_map=None, seed=None, include_snan=True):
        """
            Args:
                precision (ML_Format): floating-point format
                weight_map (dict): map category -> weigth

        """
        self.precision = precision

        # default category mapping
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
        self.sp_list = self.get_special_value_list(include_snan=False)

    @staticmethod
    def from_interval(precision, low_bound, high_bound, uniform=False):
        weight_map = {
            (FPRandomGen.Category.FPLogInterval if not uniform else FPRandomGen.Category.FPUniformInterval)(low_bound, high_bound): 1.0,
        }
        return FPRandomGen(precision, weight_map)

    def get_special_value_list(self, include_snan=True):
        """ Returns a list a special values in the generator precision """
        return  [
            sp_class(self.precision) for sp_class in
            FPRandomGen.special_value_ctor if (not sp_class is FP_SNaN or include_snan)
        ]


    def generate_sign(self):
        """ Generate a random sign value {-1.0, 1.0} """
        return sollya.SollyaObject(-1.0) if self.random.randrange(2) == 1 else \
               sollya.SollyaObject(1.0)


def get_value_exp(value):
    """ return the binary exponent of value """
    return sollya.ceil(sollya.log2(abs(value)))

class MPFPRandomGen:
    """ random generator for multi-precision floating-point numbers """
    def __init__(self, mp_format, weight_map=None):
        weight_map = normalize_map({
            FPRandomGen.Category.SpecialValues: 0.0,
            FPRandomGen.Category.Subnormal: 0.0,
            FPRandomGen.Category.Normal: 1.0,
        } if weight_map is None else weight_map)

        self.mp_format = mp_format
    
        # distribution map for lower limbs
        lower_weight_map = normalize_map({
            FPRandomGen.Category.ZeroExp: 1.0,
        })

        self.rng_gen_list = [
            FPRandomGen(mp_format.field_format_list[0], weight_map=weight_map)
        ] + [
            FPRandomGen(
                limb_format, weight_map=lower_weight_map
            ) for limb_format in mp_format.field_format_list[1:]
        ]

    @staticmethod
    def from_interval(precision, low_bound, high_bound, uniform=False):
        weight_map = {
            (FPRandomGen.Category.FPLogInterval if not uniform else FPRandomGen.Category.UniformInterval)(low_bound, high_bound): 1.0,
        }
        return MPFPRandomGen(precision, weight_map)

    def get_new_value(self):
        acc = self.rng_gen_list[0].get_new_value()
        last_exp = get_value_exp(acc)
        for index, rng in enumerate(self.rng_gen_list[1:], 1):
            mantissa_size = self.mp_format.field_format_list[index-1].get_mantissa_size()
            last_exp -= mantissa_size
            new_limb = S2**(last_exp) * rng.get_new_value()
            acc += new_limb
        return acc

def get_precision_rng(precision, value_range=None, uniform=False):
    if value_range is None:
        # default full-range value generation
        base_format = precision.get_base_format()
        if isinstance(base_format, ML_FP_MultiElementFormat):
            return MPFPRandomGen(precision)
        elif isinstance(base_format, ML_FP_Format):
            return FPRandomGen(precision, include_snan=False)
        elif isinstance(base_format, ML_Fixed_Format):
            return FixedPointRandomGen(precision.integer_size, precision.frac_size, precision.signed)
        else:
            Log.report(Log.Error, "unsupported format {}/{} in get_precision_rng", precision, base_format)
    else:
        if isinstance(value_range, UniformInterval):
            low_bound = sollya.inf(value_range.interval)
            high_bound = sollya.sup(value_range.interval)
            uniform = True
        else:
            low_bound = sollya.inf(value_range)
            high_bound = sollya.sup(value_range)
            uniform = uniform
        return get_precision_rng_with_defined_range(precision, low_bound, high_bound, uniform)

def get_precision_rng_with_defined_range(precision, inf_bound, sup_bound, uniform=False):
    """ build a random number generator for format @p precision
        which generates values within the range [inf_bound, sup_bound] """
    base_format = precision.get_base_format()
    if isinstance(base_format, ML_FP_MultiElementFormat):
        return MPFPRandomGen.from_interval(precision, inf_bound, sup_bound)
    elif isinstance(base_format, ML_FP_Format):
        return FPRandomGen.from_interval(precision, inf_bound, sup_bound, uniform)
    elif isinstance(base_format, ML_Fixed_Format):
        return FixedPointRandomGen.from_interval(precision, inf_bound, sup_bound)
    else:
        Log.report(Log.Error, "unsupported format {}/{} in get_precision_rng", precision, base_format)

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

