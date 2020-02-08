
import sollya

S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_formats import ML_Binary32
from metalibm_core.core.ml_table import ML_ApproxTable

generic_inv_approx_table = ML_ApproxTable(
    dimensions = [2**7], 
    index_size=7,
    storage_precision = ML_Binary32,
    init_data = [
        sollya.round(1/(1.0 + i * S2**-7), 9, sollya.RN) for i in range(2**7)
    ]
)

invsqrt_approx_table = ML_ApproxTable(
    dimensions = [2**8],
    index_size=8,
    storage_precision = ML_Binary32,
    init_data = [
        sollya.round(1/sollya.sqrt(1.0 + i * S2**-8), 9, sollya.RN) for i in range(2**8)
    ]
)
