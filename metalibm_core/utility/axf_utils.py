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
import json
import sollya

from metalibm_core.core.indexing import Indexing, SubIntervalIndexing
from metalibm_core.core.polynomials import Polynomial
from metalibm_core.utility.ml_template import precision_parser

#NOTES:
#  ApproxError, SimplePolyApprox, PieceWiseApprox,
#  UniformPieceWiseApprox and their derivatives
#  are metalibm-pure object, containing numerical values, metalibm's
#  format values, and metalibm's polynomial objects
#
#  AXF_<class> contains string-encoded property
#  - their constructors accept metalibm objects or str as input values
#    for basic parameter (function, format, ...) and list of parameters
#    but require AXF_<class> object as parameter for more complex
#    objects (e.g. AXF_SimplePolyApprox)
#
#
#


class SimplePolyApprox:
    """ description of a simple Polynomial approximation.
        Only the coefficient whose indexes are listed in <degree_list>
        are non-zero, <format_list> contains an ordered list of non-zero
        coefficient format """
    def __init__(self, poly, fct, degree_list, format_list, interval,
                 approx_error=None):
        self.function = fct
        self.degree_list = [int(d) for d in degree_list]
        self.format_list = [f for f in format_list]
        self.interval = interval
        self.poly = poly
        self.approx_error = approx_error

class ApproxError:
    error_type = None
    def __init__(self, value):
        self.value = value

class AbsoluteApproxError(ApproxError):
    error_type = "absolute"
    def __gt__(error1, error2):
        assert isinstance(error2, AbsoluteApproxError)
        return error1.value > error2.value

class RelativeApproxError(ApproxError):
    error_type = "relative"
    def __gt__(error1, error2):
        assert isinstance(error2, RelativeApproxError)
        return error1.value > error2.value

class AXF_ApproxError:
    """ AXF approximation error """
    def __init__(self, error_type, error_value):
        self.error_type = error_type
        self.error_value = str(error_value)

    def serialize_to_dict(self):
        return {"type": self.error_type, "value": self.error_value}


    @staticmethod
    def deserialize_from_dict(d):
        return AXF_ApproxError(d["type"], d["value"])

    def export_to_ml_error(self):
        if self.error_type == "absolute":
            return AbsoluteApproxError(sollya.parse(self.error_value))
        elif self.error_type == "relative":
            return RelativeApproxError(sollya.parse(self.error_value))
        else:
            raise NotImplementedError
    @staticmethod
    def from_AE(approx_error):
        """ import from ApproxError object """
        return AXF_ApproxError(approx_error.error_type, approx_error.value)

class UniformPieceWiseApprox:
    """ piece-wise polynomial approximation with
        uniformly divided input interval """
    def __init__(self, function, precision, interval,
                 num_intervals, max_degree, error_bound,
                 odd=False, even=False, tag="", approx_list=None, indexing=None):
        self.function = function
        self.interval = interval
        self.num_intervals = int(num_intervals)
        self.max_degree = int(max_degree)
        self.error_bound = error_bound
        self.odd = bool(odd)
        self.even = bool(even)
        self.approx_list = [] if approx_list is None else approx_list
        self.precision = precision
        self.tag = tag
        self.indexing = indexing if not indexing is None else SubIntervalIndexing(self.interval, self.num_intervals)


class GenericPolynomialSplit:
    """ generic piece-wise polynomial approximation
        with arbitrary input interval """
    def __init__(self, offset_fct, indexing, max_degree, approx_error,
                 precision, tag="", approx_list=None, odd=False, even=False):
        self.offset_fct = str(offset_fct)
        self.indexing = indexing
        assert isinstance(approx_error, ApproxError)
        self.approx_error = approx_error
        self.precision = precision
        self.approx_list = [] if approx_list is None else approx_list
        self.tag = tag
        self.max_degree = int(max_degree)
        self.odd = odd
        self.even = even


class AXF_SimplePolyApprox(yaml.YAMLObject):
    """ AXF object for basic polynomial approximation """
    yaml_tag = u'!SimplePolyApprox'
    def __init__(self, poly, fct, degree_list, format_list, interval,
                 approx_error=None):
        assert isinstance(poly, AXF_Polynomial)
        self.poly = poly
        self.function = str(fct)
        self.degree_list = [int(d) for d in degree_list]
        self.format_list = [str(f) for f in format_list]
        self.interval = str(interval)
        # TODO/FIXME: we should not have to test class of approx_error here
        #             API of AXF_SimplePolyApprox should should between them
        assert isinstance(approx_error, AXF_ApproxError)
        self.approx_error = approx_error

    def export(self):
        return yaml.dump(self)

    def serialize_to_dict(self):
        return {
            "class": self.yaml_tag,
            "function": self.function,
            "approx_params": {
                "degree_list": self.degree_list,
                "format_list": self.format_list,
            },
            "interval": self.interval,
            "approx_data": self.poly.serialize_to_dict(),
            "approx_error": self.approx_error.serialize_to_dict()
        }
    @staticmethod
    def from_SPA(simple_poly_approx):
        return AXF_SimplePolyApprox(
            AXF_Polynomial(simple_poly_approx.poly.coeff_map),
            str(simple_poly_approx.fct),
            simple_poly_approx.degree_list,
            [str(f) for f in simple_poly_approx.format_list],
            str(simple_poly_approx.interval),
            simple_poly_approx.absolute,
            AXF_ApproxError.from_AE(simple_poly_approx.approx_error),
        )

    @staticmethod
    def deserialize_from_dict(d):
        return AXF_SimplePolyApprox(
            AXF_Polynomial.deserialize_from_dict(d["approx_data"]),
            d["function"],
            d["approx_params"]["degree_list"],
            d["approx_params"]["format_list"],
            d["interval"],
            approx_error=AXF_ApproxError.deserialize_from_dict(d["approx_error"]),
        )

    def export_to_SPA(self):
        """ convert object to SimplePolyApprox """
        return SimplePolyApprox(
            self.poly.export_to_poly(), # poly
            self.function,
            self.degree_list,
            [precision_parser(f) for f in self.format_list],
            sollya.parse(self.interval),
            approx_error=self.approx_error.export_to_ml_error(),
        )

    def to_ml_object(self):
        return self.export_to_SPA()


class AXF_Polynomial(yaml.YAMLObject):
    """ AXF object for polynomial object encoding """
    yaml_tag = u'!Polynomial'
    def __init__(self, coeff_map):
        self.coeff_map = {int(k): str(v) for k,v in coeff_map.items()}
    def __repr__(self):
        return "%s(coeff_map=%r)" % (
            self.__class__.__name__, self.coeff_map)
    def export_to_poly(self):
        return Polynomial({k: sollya.parse(v) for k,v in self.coeff_map.items()})
    def convert_coeff_map(self):
        return {k: sollya.parse(v) for k,v in self.coeff_map.items()}
    @staticmethod
    def deserialize_from_dict(d):
        return AXF_Polynomial({int(k): v for k, v in d.items()})
    @staticmethod
    def from_poly(poly):
        return AXF_Polynomial(poly.coeff_map)
    def serialize_to_dict(self):
        return self.coeff_map
    def to_ml_object(self):
        return self.export_to_poly()


class AXF_UniformPiecewiseApprox(yaml.YAMLObject):
    """ AXF object for piecewise approximation encoding """
    yaml_tag = u'!PieceWiseApprox'
    def __init__(self, function, precision, interval,
                 num_intervals, max_degree, approx_error,
                 odd=False, even=False, tag="",
                 approx_list=None, indexing=None):
        self.function = str(function)
        self.interval = str(interval)
        self.indexing = str(indexing)
        self.num_intervals = int(num_intervals)
        self.max_degree = int(max_degree)
        # TODO/FIXME: we should not have to test class of approx_error here
        #             API of AXF_SimplePolyApprox should should between them
        assert isinstance(approx_error, AXF_ApproxError)
        self.approx_error = approx_error
        self.odd = bool(odd)
        self.even = bool(even)
        self.approx_list = [] if approx_list is None else approx_list
        self.precision = str(precision)
        self.tag = tag

    def export(self):
        sollya.settings.display = sollya.hexadecimal
        return yaml.dump(self)

    def serialize_to_dict(self):
        """ re-implementation of __dict__ compatible with JSON-based AXF
            emission """
        return {
            "class": self.yaml_tag,
            "function": self.function,
            "precision": self.precision,
            "interval": self.interval,
            "approx_params": {
                "indexing": self.indexing,
                "num_intervals": self.num_intervals,
                "max_degree": self.max_degree,
                "odd": self.odd, "even": self.even,
            },
            "approx_error": self.approx_error.serialize_to_dict(),
            "tag": self.tag,
            "approx_data": [approx.serialize_to_dict() for approx in self.approx_list],
        }

    @staticmethod
    def from_UPWA(upwa):
        """ build and AXF_UniformPiecewiseApprox from a
            UniformPieceWiseApprox object """
        return AXF_UniformPiecewiseApprox(
            upwa.function,
            upwa.precision,
            upwa.interval,
            upwa.num_intervals,
            upwa.max_degree,
            AXF_ApproxError.from_AE(upwa.error_bound),
            upwa.odd,
            upwa.even,
            upwa.tag,
            approx_list=[AXF_SimplePolyApprox.from_SPA(spa) for spa in upwa.approx_list],
            indexing=upwa.indexing
        )

    @staticmethod
    def deserialize_from_dict(d):
        return AXF_UniformPiecewiseApprox(
            d["function"],
            d["precision"],
            d["interval"],
            d["approx_params"]["num_intervals"],
            d["approx_params"]["max_degree"],
            AXF_ApproxError.deserialize_from_dict(d["approx_error"]),
            odd=d["approx_params"]["odd"], even=d["approx_params"]["even"],
            tag=d["tag"],
            approx_list=[AXF_SimplePolyApprox.deserialize_from_dict(v) for v in d["approx_data"]],
            indexing=d["approx_params"]["indexing"]
        )

    def export_to_UPWA(self):
        """ convert self AXF_UniformPiecewiseApprox object
            to a UniformPieceWiseApprox object """
        return UniformPieceWiseApprox(
            self.function,
            precision_parser(self.precision),
            sollya.parse(self.interval),
            self.num_intervals,
            self.max_degree,
            self.approx_error.export_to_ml_error(),
            self.odd,
            self.even,
            self.tag,
            approx_list=[(axf_spa.export_to_SPA()) for axf_spa in self.approx_list],
            indexing=Indexing.parse(self.indexing)
        )

    def to_ml_object(self):
        return self.export_to_UPWA()


class AXF_GenericPolynomialSplit(yaml.YAMLObject):
    """ AXF object for a piecewise generic polynomial approximation encoding """
    yaml_tag = u'!GenericPolynomialSplit'
    def __init__(self, offset_fct, precision, interval, indexing, max_degree, approx_error, tag="", approx_list=None, odd=False, even=False):
        self.offset_fct = str(offset_fct)
        self.indexing = str(indexing)
        self.interval = str(interval)
        assert isinstance(approx_error, AXF_ApproxError)
        self.approx_error = approx_error
        self.precision = str(precision)
        self.approx_list = [] if approx_list is None else approx_list
        self.tag = tag
        self.max_degree = int(max_degree)
        self.odd = odd
        self.even = even

    def export(self):
        sollya.settings.display = sollya.hexadecimal
        return yaml.dump(self)

    def serialize_to_dict(self):
        """ re-implementation of __dict__ compatible with JSON-based AXF emission """
        return {
            "class": self.yaml_tag,
            # TODO/FIXME: "function" field
            "precision": self.precision,
            "approx_params": {
                "offset_fct": self.offset_fct,
                "indexing": self.indexing,
                "max_degree": self.max_degree,
                "odd": self.odd,
                "even": self.even,
            },
            "interval": self.interval,
            "approx_error": self.approx_error.serialize_to_dict(),
            "approx_data": [approx.serialize_to_dict() for approx in self.approx_list],
            "tag": self.tag,
        }

    @staticmethod
    def from_GenericPolynomialSplit(gps):
        """ build and AXF_GenericPolynomialSplit from a
            GenericPolynomialSplit object """
        return AXF_GenericPolynomialSplit(
            gps.offset_fct,
            gps.precision,
            gps.interval,
            gps.indexing,
            gps.max_degree,
            AXF_ApproxError.from_AE(gps.approx_error),
            gps.tag,
            approx_list=[AXF_SimplePolyApprox.from_SPA(spa) for spa in upwa.approx_list],
            odd=gps.odd,
            even=gps.even
        )

    @staticmethod
    def deserialize_from_dict(d):
        return AXF_GenericPolynomialSplit(
            d["approx_params"]["offset_fct"],
            d["precision"],
            d["interval"],
            d["approx_params"]["indexing"],
            d["approx_params"]["max_degree"],
            AXF_ApproxError.deserialize_from_dict(d["approx_error"]),
            d["tag"],
            approx_list=[AXF_SimplePolyApprox.deserialize_from_dict(v) for v in d["approx_data"]],
            odd=d["approx_params"]["odd"],
            even=d["approx_params"]["even"]
        )

    def export_to_GPS(self):
        """ convert self AXF_GenericPolynomialSplit object
            to a GenericPolynomialSplit object """
        return GenericPolynomialSplit(
            self.offset_fct,
            Indexing.parse(self.indexing),
            self.max_degree,
            self.approx_error.export_to_ml_error(),
            precision_parser(self.precision),
            self.tag,
            approx_list=[(axf_spa.export_to_SPA()) for axf_spa in self.approx_list],
            odd=self.odd,
            even=self.even
        )

    def to_ml_object(self):
        """ generic AXF class to Metalibm Object transform API """
        return self.export_to_GPS()


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
# yaml.add_constructor(u'!Polynomial', poly_constructor)

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
        return yaml.load(s).to_ml_object()

class AXF_JSON_Exporter:
    @staticmethod
    def to_file(filename, serialized_approx):
        with open(filename, "w") as out_stream:
            out_stream.write(json.dumps(serialized_approx, sort_keys=True, indent=4))

class AXF_JSON_Importer:
    """ Import for json-based AXF storage to Metalibm's classes """
    @staticmethod
    def from_file(filename):
        """ import an approximation description from a source file in .axf
            format """
        with open(filename, 'r') as stream:
            return AXF_JSON_Importer.from_str(stream.read())

    @staticmethod
    def from_str(s):
        """ import an approximation description from a string description
            in AXF format """
        axf_dict = json.loads(s)
        # json AXF string contains ether an approximation as a dict
        # or a list of approximations as a list of dict
        if isinstance(axf_dict, list):
            return [AXF_JSON_Importer.serialized_dict_to_ml_object(sub_dict) for sub_dict in axf_dict]
        else:
            return AXF_JSON_Importer.serialized_dict_to_ml_object(axf_dict)

    @staticmethod
    def serialized_dict_to_ml_object(serialized_dict):
        for ctor_class in [AXF_UniformPiecewiseApprox, AXF_GenericPolynomialSplit]:
            if ctor_class.yaml_tag == serialized_dict["class"]:
                return ctor_class.deserialize_from_dict(serialized_dict).to_ml_object()
        raise Exception("unable to find deserializer for json class %s" % serialized_dict["class"])

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
