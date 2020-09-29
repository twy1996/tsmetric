import pytest
import numpy as np

import metrics


def test_dtw():
    series1 = np.array([1, 2, 3])
    series2 = np.array([2, 2, 2, 3, 4])
    distance = metrics.DTW(series1, series2)
    assert distance == 2

def test_ddtw():
    series1 = np.array([1, 2, 3])
    series2 = np.array([2, 2, 2, 3, 4])
    
    diff_series1 = np.array([1])
    diff_series2 = np.array([0, 0.25, 1])

    assert (metrics.DDTW(series1, series2) == 
            metrics.DTW(diff_series1, diff_series2))
    

if __name__ == '__main__':
    
    pytest.main([__file__])
