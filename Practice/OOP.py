




class Person(object):
    def __init__(self, name):
        self.name = name

    def reveal_identity(self):
        print('My name is {}'.format(self.name))

class SuperHero(Person):
    def __int__(self, name, hero_name):
        super().__init__(name)
        self.hero_name = hero_name

    def reveal_identity(self):
        super().reveal_identity()
        print("... And I am {}".format(self.hero_name))


Stefan = Person("Stefan")
Stefan.reveal_identity()

wade = SuperHero("wade", 'deadpool')
wade.reveal_identity()

















