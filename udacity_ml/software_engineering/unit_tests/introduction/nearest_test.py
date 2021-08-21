from udacity_ml.software_engineering.unit_tests.introduction.nearest import nearest_square

# slightly better
print(f'Nearest square <= 5: returned {nearest_square(5)}, expected: 4.')
print(f'Nearest square <= -12: returned {nearest_square(-12)}, expected: 0.')
print(f'Nearest square <= 9: returned {nearest_square(9)}, expected: 9.')
print(f'Nearest square <= 23: returned {nearest_square(23)}, expected: 16.')

# slightly better [2]
assert nearest_square(5) == 4
assert nearest_square(-12) == 0
assert nearest_square(9) == 9
assert nearest_square(23) == 16



