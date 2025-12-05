import numpy as np
import pandas as pd
from ospa2 import OSPA2

def test_ospa2_basic():
    gt = pd.DataFrame({
        'ts': [0,1,2,0,1,2],
        'track_id': [1,1,1,2,2,2],
        'x': [0,0.1,0.2, 5,5.1,5.2],
        'y': [0,0.0,0.1, 5,5.0,5.1]
    })
    trk = pd.DataFrame({
        'ts': [0,1,2,0,1,2],
        'track_id': [1,1,1,2,2,2],
        'x': [0,0.1,0.2, 5,5.1,5.2],
        'y': [0,0.0,0.1, 5,5.0,5.1]
    })
    o = OSPA2(c=100,p=1,q=1,window_length=3,cols=['x','y'],id_col='track_id')
    ts, o2, loc, card = o.ospa2_over_time(gt, trk)
    assert all(abs(v) < 1e-12 for v in o2)


