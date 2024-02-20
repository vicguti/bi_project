class Dog:

    tricks = []             # mistaken use of a class variable

    def __init__(self, name):
        self.name = name

    def add_trick(self, trick):
        self.tricks.append(trick)



print(Dog("Fido").name)


class test:

    def __init__(self, test_value):
        self.value = test_value

    def print_value(self, value):
        return self.value + value


class testing_test(test):

    def __init__(self, var_1, test_value):
        self.var_2 = var_1
        super(testing_test, self).__init__(test_value=test_value)

    def return_value(self):
        return self.var_2

print(testing_test("hola", "wow").return_value())

print(testing_test("hola", "wow").print_value("test"))



