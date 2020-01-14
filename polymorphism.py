"""
    多态：子类在继承了父类的方法后，可以对父类已有的方法给出新的实现版本，这个动作称之为方法重写（override）。
    通过方法重写我们可以让父类的同一个行为在子类中拥有不同的实现版本，当我们调用这个经过子类重写的方法时，不同
    的子类对象会表现出不同的行为，这个就是多态（poly-morphism）。
"""

from abc import ABCMeta, abstractmethod


class Pet(object, metaclass = ABCMeta):
    """
    Pet类作为一个抽象类，抽象类不能够创建对象，它的存在是为了让其他类继承他。通过abc模块的ABCMeta元类和abstractmethod包装器来达
    到抽象类的效果。
    """

    def __init__(self, nickname):
        self._nickname = nickname

    @abstractmethod
    def make_voice(self):
        """发出声音"""
        pass


class Dog(Pet):
    """狗"""

    def make_voice(self):
        print('%s: 汪汪汪...' % self._nickname)


class Cat(Pet):
    """猫"""

    def make_voice(self):
        print('%s: 喵喵喵' % self._nickname)


def main():
    pets = [Dog('旺财'), Cat('凯蒂'), Dog('大黄')]
    for pet in pets:
        pet.make_voice()


if __name__ == '__main__':
    main()
