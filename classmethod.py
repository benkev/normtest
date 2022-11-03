class Student:
    name = 'unknown' # class attribute
    def __init__(self):
        self.age = 20  # instance attribute

    @classmethod
    def tostring(cls):
        print('Student Class Attribute: name = ', cls.name)

    def tostring1(self):
        print('Student Instance Attribute: age = ', self.age)

    def tostring2(self):
        print('Student Attributes: age, name = ', self.age, ', ', self.name)

    
    
