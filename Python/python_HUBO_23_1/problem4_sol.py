class Person():
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def say_name(self):
        print(self.name)
    def say_age(self):
        print(self.age)
    def get_old_oneyear(self):
        self.age+=1
        print(self.age)



class Student(Person):
    def __init__(self,name,age,major,GPA):
        self.name=name
        self.age=age
        self.major=major
        self.GPA=GPA
    def study(self):
        if self.GPA<4.5:
            self.GPA+=0.1
        print(self.GPA)
    def say_name(self):
        super().say_name()
        super().say_age()
        print(self.major)
        print(self.GPA)

person1 = Student('Alice', 25, 'INCE' ,3.0)
person2 = Student('Bob', 30, 'EE' ,  4.4)
person1.study()
person1.study()
person2.study()
person2.study()
person1.say_name()
