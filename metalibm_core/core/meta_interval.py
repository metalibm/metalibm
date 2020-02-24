from sollya import Interval, SollyaObject
import sollya
import itertools

def convert_to_MetaInterval(obj):
    """ convert basic numeric objects to MetaInterval if possible """
    if isinstance(obj, MetaInterval):
        return obj
    elif isinstance(obj, SollyaObject) and obj.is_range():
        return MetaInterval(obj)
    elif isinstance(obj, (float, int, SollyaObject)):
        return MetaInterval(lhs=obj)
    else:
        raise NotImplementedError

def convert_to_MetaIntervalList(obj):
    """ convert basic numeric object to MetaIntervalList if possible """
    if isinstance(obj, MetaIntervalList):
        return obj
    elif isinstance(obj, MetaInterval):
        return MetaIntervalList([obj])
    elif isinstance(obj, (float, int, SollyaObject)):
        return MetaIntervalList([convert_to_MetaInterval(obj)])
    else:
        raise NotImplementedError

class MetaInterval:
    """ Metalibm numerical contiguous interval object """
    def __init__(self, interval=None, lhs=None, rhs=None):
        if not interval is None:
            self.interval = interval
        elif not lhs is None:
            if not rhs is None:
                self.interval = Interval(lhs, rhs)
            else:
                self.interval = Interval(lhs)
        else:
            # empty interval
            self.interval = None

    def __or__(lhs, rhs):
        """ interval union """
        lhs = convert_to_MetaInterval(lhs)
        rhs = convert_to_MetaInterval(rhs)
        if lhs.sup < rhs.inf:
            return MetaIntervalList([lhs, rhs])
        elif rhs.sup < lhs.inf:
            return MetaIntervalList([rhs, lhs])
        else:
            return MetaInterval(lhs=min(lhs.inf, rhs.inf), rhs=max(lhs.sup, rhs.sup))
    def __and__(lhs, rhs):
        """ interval intersection """
        lhs = convert_to_MetaInterval(lhs)
        rhs = convert_to_MetaInterval(rhs)
        if lhs.sup < rhs.inf or rhs.sup < lhs.inf:
            return MetaInterval(None)
        else:
            return MetaInterval(lhs=max(lhs.inf, rhs.inf), rhs=min(lhs.sup, rhs.sup))


    def __add__(lhs, rhs):
        if isinstance(rhs, MetaIntervalList):
            return MetaIntervalList.__add__(MetaIntervalList([lhs]), rhs)
        elif lhs.interval is None or rhs.interval is None:
            return MetaInterval(None)
        else:
            lhs = convert_to_MetaInterval(lhs)
            rhs = convert_to_MetaInterval(rhs)
            return MetaInterval(interval=lhs.interval + rhs.interval)
            # return MetaInterval(lhs=lhs.inf+rhs.inf, rhs=lhs.sup+rhs.sup)
    def __sub__(lhs, rhs):
        if isinstance(rhs, MetaIntervalList):
            return MetaIntervalList.__sub__(MetaIntervalList([lhs]), rhs)
        elif lhs.interval is None or rhs.interval is None:
            return MetaInterval(None)
        else:
            lhs = convert_to_MetaInterval(lhs)
            rhs = convert_to_MetaInterval(rhs)
            return  MetaInterval(interval=lhs.interval - rhs.interval)
            #MetaInterval(lhs=min(lhs.inf - rhs.sup, rhs.inf - lhs.sup), rhs=max(lhs.sup - rhs.inf, rhs.sup - lhs.inf))
    def __mul__(lhs, rhs):
        if isinstance(rhs, MetaIntervalList):
            return MetaIntervalList.__mul__(MetaIntervalList([lhs]), rhs)
        elif lhs.interval is None or rhs.interval is None:
            return MetaInterval(None)
        else:
            lhs = convert_to_MetaInterval(lhs)
            rhs = convert_to_MetaInterval(rhs)
            return MetaInterval(interval=lhs.interval * rhs.interval)
            #extrema_list = [
            #    lhs.inf * rhs.inf,
            #    lhs.inf * rhs.sup,
            #    lhs.sup * rhs.inf,
            #    lhs.sup * rhs.sup,
            #]
            #return MetaInterval(lhs=min(extrema_list), rhs=max(extrema_list))
    def __truediv__(lhs, rhs):
        if isinstance(rhs, MetaIntervalList):
            MetaIntervalList.__div__(MetaIntervalList([lhs.interval]), rhs)
        elif isinstance(rhs, (int, float)):
            rhs = MetaInterval(Interval(rhs))
        elif lhs.interval is None or rhs.interval is None:
            return MetaInterval(None)
        lhs = convert_to_MetaInterval(lhs)
        rhs = convert_to_MetaInterval(rhs)
        return MetaInterval(interval=lhs.interval / rhs.interval)

    def __neg__(self):
        if self.interval is None:
            return self
        return MetaInterval(interval=-self.interval)

    def __contains__(self, value):
        if self.interval is None:
            return False
        if isinstance(value, (int, float)):
            return value >= self.inf and value <= self.sup
        elif isinstance(value, SollyaObject):
            if value.is_range():
                inf(value) >= self.inf and sup(value) <= self.sup
            else:
                return value >= self.inf and value <= self.sup
        elif isinstance(value, MetaInterval):
            return value.interval in self
        elif isinstance(value, MetaIntervalList):
            return any(sub in self for sub in value.interval_list)
        else: 
            raise NotImplementedError

    def __repr__(self):
        if self.interval is None:
            return "[empty]"
        return "[{};{}]".format(self.inf, self.sup)

    @property
    def sup(self):
        return sup(self.interval)
    @property
    def inf(self):
        return inf(self.interval)

    @property
    def is_empty(self):
        return self.interval == None

def refine_interval_list(interval_list):
    """ canonize internal interval_list to contains
        sorted, disjoint intervals """
    sorted_list = sorted(interval_list, key=lambda i: i.inf)
    refined_list = [sorted_list[0]]
    for sub_interval in sorted_list[1:]:
        if sub_interval.inf <= refined_list[-1].sup:
            # merge interval
            refined_list[-1] = refined_list[-1] | sub_interval
        else:
            refined_list.append(sub_interval)
    return refined_list


class MetaIntervalList:
    """ extended interval object which can store
        union of disjoint intervals """
    def __init__(self, interval_list):
        # refine on creation => allow to easily retrieve inf and sup bound
        # as interval order will not be modified
        self._interval_list = refine_interval_list(list(interval_list))
    
    @property
    def interval_list(self):
        """ specific getter for interval_list to avoid defining a setter """
        return self._interval_list

    @property
    def inf(self):
        return self._interval_list[0].inf
    @property
    def sup(self):
        return self._interval_list[-1].sup
    @property
    def is_empty(self):
        return all(sub.is_empty for sub in self.interval_list)

    def __contains__(self, value):
        return any(value in interval for interval in self.interval_set)

    def __repr__(self):
        return " \/ ".join("{}".format(interval) for interval in self.interval_list)

    def __add__(lhs, rhs):
        rhs = convert_to_MetaIntervalList(rhs)
        result = MetaIntervalList((lhs_sub + rhs_sub) for lhs_sub, rhs_sub in itertools.product(lhs.interval_list, rhs.interval_list))
        return result
    def __sub__(lhs, rhs):
        rhs = convert_to_MetaIntervalList(rhs)
        result = MetaIntervalList((lhs_sub - rhs_sub) for lhs_sub, rhs_sub in itertools.product(lhs.interval_list, rhs.interval_list))
        return result
    def __mul__(lhs, rhs):
        rhs = convert_to_MetaIntervalList(rhs)
        result = MetaIntervalList((lhs_sub * rhs_sub) for lhs_sub, rhs_sub in itertools.product(lhs.interval_list, rhs.interval_list))
        return result
    def __truediv__(lhs, rhs):
        rhs = convert_to_MetaIntervalList(rhs)
        result = MetaIntervalList((lhs_sub / rhs_sub) for lhs_sub, rhs_sub in itertools.product(lhs.interval_list, rhs.interval_list))
        return result
    def __rtruediv__(rhs, lhs):
        lhs = convert_to_MetaIntervalList(lhs)
        result = MetaIntervalList((lhs_sub / rhs_sub) for lhs_sub, rhs_sub in itertools.product(lhs.interval_list, rhs.interval_list))
        return result
    def __neg__(self):
        return MetaIntervalList(-sub for sub in self.interval_list)


def inf(obj):
    """ generic getter for interval inferior bound """
    if isinstance(obj, SollyaObject) and obj.is_range():
        return sollya.inf(obj)
    elif isinstance(obj, (MetaInterval, MetaIntervalList)):
        return obj.inf
    else:
        raise NotImplementedError


def sup(obj):
    """ generic getter for interval superior bound """
    if isinstance(obj, SollyaObject) and obj.is_range():
        return sollya.sup(obj)
    elif isinstance(obj, (MetaInterval, MetaIntervalList)):
        return obj.sup
    else:
        raise NotImplementedError

if __name__ == "__main__":
    int0 = MetaInterval(lhs=1, rhs=2)
    int1 = MetaInterval(interval=Interval(-2, 1.5))
    int2 = MetaInterval(lhs=3, rhs=4)
    int3 = int2 | int1
    print("int0: ", int0)
    print("int1: ", int1)
    print("int0 | int1:", int0 | int1)
    print("int0 & int1:", int0 & int1)
    print("int2 & int1:", int2 & int1)
    print("int2 | int1:", int3)
    print("{} in {}: {}".format(1.5, int0, 1.5 in int0))
    print("{} in {}: {}".format(1.5, int2, 1.5 in int2))
    for lhs in [int0, int1, int2, int3]:
        for rhs in [int0, int1, int2, int3]:
            print("({}) * ({}) = {}".format(lhs, rhs, lhs * rhs))
