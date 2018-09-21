class TestClass:

    'This is a static variable'
    class_name = 'TestClass'

    'Instance Variables defined in the Constrctor'
    def __init__(self, name):
        self.name = name
        print("Constructor")

    'Method without arguments'
    def a(self):
        print ("Name", self.name, sep="=")

    'Method with 2 arguments'
    def mult(self, a=1, b=1):
        return a*b

class TestClassExt (TestClass):
    def __init__(self):
        self.name = ""

tc = TestClass("Gil")
tc.a()
result = tc.mult(a=2,b=3)
print(result)

'Function reference'
_2_arg_function = tc.mult
result = _2_arg_function(5,6)
print(result)

tc2 = TestClassExt()
result = tc2.mult(7,8)
print(result)
