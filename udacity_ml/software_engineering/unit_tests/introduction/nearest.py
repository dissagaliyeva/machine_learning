def nearest_square(num):
    """Return the nearest perfect square that is less than or equal to num"""
    root = 0
    while (root + 1)**2 <= num:
        root += 1
    return root**2


# UNRELIABLE & BAD: try out different cases
# NEVER use tests in the same file
# ALWAYS define the expected answers
print(nearest_square(5))
print(nearest_square(-12))
print(nearest_square(10))
print(nearest_square(170))
print(nearest_square(0))

