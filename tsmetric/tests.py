import pytest
import numpy as np

import metrics

def test_euclidean():
    series1 = np.array([1, 2, 3])
    series2 = np.array([2, 2, 1])
    distance = metrics.Euclidean(series1, series2)
    
    assert distance == 5

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

def test_lcss():
    epsilon = 1e-1
    series1 = np.array([1, 2, 2])
    series2 = np.array([2, 2, 2, 3, 4])
    distance = metrics.LCSS(series1, series2, epsilon)
    
    assert distance == 0.6
    
if __name__ == '__main__':
    
    pytest.main([__file__])
