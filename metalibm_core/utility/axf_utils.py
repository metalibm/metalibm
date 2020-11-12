# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2020 Kalray
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
# create:           Nov  8th, 2020
# last-modified:    Nov  8th, 2020
###############################################################################

import yaml
import sollya

from metalibm_core.core.polynomials import Polynomial

class AXF_SimplePolyApprox(yaml.YAMLObject):
    """ AXF object for basic polynomial approximation """
    yaml_tag = u'!SimplePolyApprox'
    def __init__(self, poly, fct, degree_list, format_list, interval, absolute=True, approx_error=None):
        self.function = str(fct)
        self.degree_list = [int(d) for d in degree_list]
        self.format_list = [str(f) for f in format_list]
        self.interval = str(interval)
        self.absolute = absolute
        self.poly = poly
        self.approx_error = str(approx_error)

    def export(self):
        return yaml.dump(self)


class AXF_Polynomial(yaml.YAMLObject):
    """ AXF object for polynomial object encoding """
    yaml_tag = u'!Polynomial'
    def __init__(self, coeff_map):
        self.coeff_map = {int(k): str(v) for k,v in coeff_map.items()}
    def __repr__(self):
        return "%s(coeff_map=%r)" % (
            self.__class__.__name__, self.coeff_map)
    def to_ml_poly(self):
        return Polynomial({k: sollya.parse(v) for k,v in self.coeff_map.items()})
    def convert_coeff_map(self):
        return {k: sollya.parse(v) for k,v in self.coeff_map.items()}

class AXF_UniformPiecewiseApprox(yaml.YAMLObject):
    """ AXF object for piecewise approximation encoding """
    yaml_tag = u'!PieceWiseApprox'
    def __init__(self, function, precision, bound_low, bound_high,
                 num_intervals, max_degree, error_threshold,
                 odd=False, even=False, tag=""):
        self.function = str(function)
        self.bound_low = str(bound_low)
        self.bound_high = str(bound_high)
        self.num_intervals = int(num_intervals)
        self.max_degree = int(max_degree)
        self.error_threshold = str(error_threshold)
        self.odd = bool(odd)
        self.even = bool(even)
        self.approx_list = []
        self.precision = str(precision)
        self.tag = tag

    def export(self):
        sollya.settings.display = sollya.hexadecimal
        return yaml.dump(self)


class AXF_GenericPolynomialSplit(yaml.YAMLObject):
    """ AXF object for a piecewise generic polynomial approximation encoding """
    yaml_tag = u'!GenericPolynomialSplit'
    def __init__(self, offset_fct, indexing, poly_max_degree, target_eps, coeff_precision, tag=""):
        self.offset_fct = str(offset_fct)
        self.indexing = str(indexing)
        self.target_eps = str(target_eps)
        self.coeff_precision = str(coeff_precision)
        self.approx_list = []
        self.tag = tag
        self.poly_max_degree = int(poly_max_degree)

    def export(self):
        sollya.settings.display = sollya.hexadecimal
        return yaml.dump(self)

# add specific YAML representation for metalibm's Polynomial
def poly_representer(dumper, data):
    return AXF_Polynomial.to_yaml(dumper, AXF_Polynomial(data.coeff_map))

def poly_constructor(loader, node):
    """ yaml loader for !Polynomial to Polynomial class  """
    instance = Polynomial.__new__(Polynomial)
    yield instance
    state = loader.construct_mapping(node, deep=True)
    instance.__init__({k: sollya.parse(str(v)) for k,v in state["coeff_map"].items()})

yaml.add_representer(Polynomial, poly_representer)
yaml.add_constructor(u'!Polynomial', poly_constructor)

class AXF_Exporter:
    def __init__(self, fct, approx_list):
        self.fct = fct
        self.approx_list = approx_list

    def export(self):
        return yaml.dump({"fct": self.fct, "subapproximation": self.approx_list})


class AXF_Importer:
    """ Import for AXF storage to Metalibm's classes """
    def __init__(self, stream):
        self.content = yaml.load(stream, Loader=yaml.Loader)

    @staticmethod
    def from_file(filename):
        """ import an approximation description from a source file in .axf
            format """
        with open(filename, 'r') as stream:
            return AXF_Importer.from_str(stream.read())

    @staticmethod
    def from_str(s):
        """ import an approximation description from a string description
            in AXF format """
        return yaml.load(s)

if __name__ == "__main__":
    from metalibm_core.core.polynomials import Polynomial
    import sollya
    sollya.settings.display = sollya.hexadecimal
    axf_export = AXF_Exporter("exp(x)", Polynomial({0: 1, 1: sollya.round(sollya.exp(1), sollya.binary64, sollya.RN)}))
    stream = axf_export.export()
    print(stream)
    axf_import = AXF_Importer(stream)
    print(axf_import.content)
    print(axf_import.content['subapproximation'])
