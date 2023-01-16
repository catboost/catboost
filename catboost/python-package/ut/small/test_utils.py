from catboost.utils import calculate_quantization_grid


def test_quantization_calculator():
    assert calculate_quantization_grid([1, 2], 2) == [1.5]
    assert calculate_quantization_grid([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], 1, 'Median') == []
    assert calculate_quantization_grid([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], 1, 'GreedyLogSum') == [1.5]
    assert calculate_quantization_grid([11, 1, 1, 1, 1, 1, 1, 2], 2, 'GreedyLogSum') == [1.5, 6.5]
