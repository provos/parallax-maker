import unittest
from . import constants as C


class TestConstants(unittest.TestCase):

    def test_unique_constants(self):
        constants_dict = {
            name: value for name, value in vars(C).items() if isinstance(value, str)
        }
        values = list(constants_dict.values())
        unique_values = set(values)

        if len(values) != len(unique_values):
            # Find non-unique values
            duplicates = set([value for value in values if values.count(value) > 1])
            non_unique_constants = {
                name: value
                for name, value in constants_dict.items()
                if value in duplicates
            }
            self.fail(f"String constants are not unique: {non_unique_constants}")


if __name__ == "__main__":
    unittest.main()
