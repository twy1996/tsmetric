import numpy as np


def Euclidean(series1, series2):
    series1 = series1.reshape(-1)
    series2 = series2.reshape(-1)
    m, n = series1.shape[0], series2.shape[0]
    assert m == n  # special case
    diff = series1 - series2
    
    return np.matmul(diff.reshape(1, -1), diff.reshape(-1, 1))
    
def DTW(series1, series2):
    """
    Dynamic Time Wraping (DTW) distance, implemented using dynamic programming
    with state compression.
    
    """
    series1 = series1.reshape(-1)
    series2 = series2.reshape(-1)
    m, n = series1.shape[0], series2.shape[0]
    
    prev = [float('inf') for _ in range(n+1)]
    prev[0] = 0
    
    for i in range(1, m+1):
        cur = [float('inf') for _ in range(n+1)]
        for j in range(1, n+1):
            cost = abs(series1[i-1] - series2[j-1])
            cur[j] = cost + min(cur[j-1], prev[j], prev[j-1])
        prev = cur
    
    return prev[-1]

def DDTW(series1, series2):
    """
    Derivative dynamic time wrping (DDTW) distance, implemented using sliding
    window with fixed size and the DTW function.
    
    """
    
    def _sliding_window(series):
        l = series.shape[0]
        if l < 3:
            msg = ('The length of series should be greater than 2 but got'
                   ' length = {}.')
            raise RuntimeError(msg.format(l))
        
        derivative_series = np.zeros((l-2))
        for i in range(l-2):
            derivative_series[i] += 0.5 * (series[i+1] - series[i] +
                                           (series[i+2] - series[i]) / 2)
        return derivative_series
    
    series1 = series1.reshape(-1)
    series2 = series2.reshape(-1)
    
    diff_series1 = _sliding_window(series1)
    diff_series2 = _sliding_window(series2)
    
    return DTW(diff_series1, diff_series2)

def WDTW(series1, series2, wmax=1, g=1):
    """
    Weighted dynamic time wraping (WDTW) distance with logistic weight.
    
    """

    def _logistic_weight(a, wmax, g, m):
        # w(a) = \frac{wmax}{1+\exp{-g \times (a-\frac{m}{2})}}
        
        assert wmax > 0
        assert m > 0
        weight =  wmax / (1 + np.exp(-g * (a -m/2)))
        return weight

    series1 = series1.reshape(-1)
    series2 = series2.reshape(-1)
    m, n = series1.shape[0], series2.shape[0]
    max_length = max(m, n)
    
    prev = [float('inf') for _ in range(n+1)]
    prev[0] = 0
    
    for i in range(1, m+1):
        cur = [float('inf') for _ in range(n+1)]
        for j in range(1, n+1):
            weight = _logistic_weight(abs(i-j), wmax, g, max_length)
            cost = weight * abs(series1[i-1] - series2[j-1])
            cur[j] = cost + min(cur[j-1], prev[j], prev[j-1])
        prev = cur
    
    return prev[-1]

def LCSS(series1, series2, epsilon=1):
    """
    Longest common sub-sequence (LCSS) distance, implemented using dynamic 
    programming with state compression. The epsilon defines the threshold for 
    two real values to be considered as the same.
    
    """
    series1 = series1.reshape(-1)
    series2 = series2.reshape(-1)
    m, n = series1.shape[0], series2.shape[0]
    max_length = max(m, n)
    
    prev = [0 for _ in range(n+1)]
    prev[0] = 0
    
    for i in range(1, m+1):
        cur = [0 for _ in range(n+1)]
        for j in range(1, n+1):
            cost = abs(series1[i-1] - series2[j-1])
            if cost <= epsilon:
                cur[j] = 1 + prev[j-1]
            else:
                cur[j] = max(cur[j-1], prev[j])
        prev = cur
    
    return 1 - prev[-1] / max_length


def TWE(series1, series2):
    """
    Time wrap edit (TWE) distance, implemented using dynamic programming with
    state compression.

    """
    
    

def MSM(series1, series2, cost=1):
    """
    Move-split-merge (MSM) distance, implemented using dynamic programming with
    state compression.
    """
    
    def _cost(new, x, y, cost):
        if (new >= x and new <= y) or (new <= x and new >= y):
            return cost
        else:
            return cost + min(abs(new - x), abs(new - y))
    
    series1 = series1.reshape(-1)
    series2 = series2.reshape(-1)
    m, n = series1.shape[0], series2.shape[0]
    
    prev = [0 for _ in range(n)]
    prev[0] = abs(series1[0] - series2[0])
    for i in range(1, n):
        prev[i] = prev[i-1] + _cost(series2[i], series1[0], series2[i-1], cost)
    
    for i in range(1, m):
        cur = [0 for _ in range(n)]
        cur[0] = prev[0] + _cost(series1[i], series1[i-1], series2[0], cost)
        for j in range(1, n):
            dist_1 = prev[j-1] + abs(series1[i] - series2[j])
            dist_2 = prev[j] + _cost(series1[i], series1[i-1], series2[j], cost)
            dist_3 = cur[j-1] + _cost(series2[j], series1[i], series2[j-1], cost)
            cur[j] = min(dist_1, dist_2, dist_3)
        prev = cur
    
    return prev[-1]


if __name__ == '__main__':
    
    series1 = np.array([1])
    series2 = np.array([0, 0.25, 1])
    print(WDTW(series1, series2))
    