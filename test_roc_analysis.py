import pytest

from main import calculate_distance, get_shorter_distance, get_x_y


@pytest.mark.parametrize(
    "data,expected",
    [
        ((440, 18, 14, 227), (0.058091286, 0.96069869),),
        ((446, 12, 22, 219), (0.091286307, 0.973799127)),
        ((444, 14, 8, 223), (0.034632035, 0.969432314)),
        ((192, 7, 15, 501), (0.02907, 0.964824)),
        (
            (192, 2, 3, 2, 5, 154, 3, 55, 6, 3, 206, 3, 4, 62, 5, 141),
            (0.06494, 0.821138),
        ),
        ((4, 6, 0, 0), (1, 0.39999999996)),
    ],
)
def test_get_x_y(data, expected):
    x_result, y_result = get_x_y(data)
    assert x_result == pytest.approx(expected[0], 1e-5)
    assert y_result == pytest.approx(expected[1], 1e-5)


@pytest.mark.parametrize(
    "data,expected",
    [
        ((0.058091286, 0.96069869), 0.070136941),
        ((0.091286307, 0.973799127), 0.094971973),
        ((0.034632035, 0.969432314), 0.046192653),
    ],
)
def test_get_distance(data, expected):
    assert calculate_distance(*data) == pytest.approx(expected, 1e-7)


def test_get_shorter_distance():
    data = (
        ("RNA", (440, 18, 14, 227)),
        ("KNN", (446, 12, 22, 219)),
        ("SVM", (444, 14, 8, 223)),
    )
    result = get_shorter_distance(data)
    assert result[0] == "SVM"
    assert result[1] == pytest.approx(0.046192653, rel=1e-8)
