import sys, os
import atexit
#import warnings

class Student:
    name = 'Beauty' # class attribute
    def __init__(self):
        self.age = 20  # instance attribute

    @classmethod
    def tostring(cls):
        print('Student Class Attribute: name = ', cls.name)

    def tostring1(self):
        print('Student Instance Attribute: age = ', self.age)

    def tostring2(self):
        print('Student Attributes: age, name = ', self.age, ', ', self.name)

    
print(__name__)

# if __name__ != "__main__":
#     import warnings
#     warnings.filterwarnings("ignore")
#     raise SystemExit

if __name__ == "__main__":

    from test_class_method import Student as S

    # Call classmethod
    S.tostring()

    s = S()

    # Call class method and instance methods
    s.tostring() 
    s.tostring1() 
    s.tostring2() 
    

