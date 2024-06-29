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

person1=Person("Alice",25)
person2=Person("Bob",30)
person1.say_name()
person1.say_age()
person2.say_name()
person2.say_age()
person1.get_old_oneyear()
person1.get_old_oneyear()
person2.get_old_oneyear()
person2.get_old_oneyear()