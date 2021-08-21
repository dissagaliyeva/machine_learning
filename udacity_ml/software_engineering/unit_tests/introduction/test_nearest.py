# Unit test with pytest
# Run "pytest" in the CLI & use "test_X" for the file name as well as function declaration!


from udacity_ml.software_engineering.unit_tests.introduction.nearest import nearest_square


def test_nearest_square_5():
    assert nearest_square(5) == 4


def test_nearest_square_9():
    assert nearest_square(9) == 9


def test_nearest_square_neg12():
    assert nearest_square(-12) == 0




