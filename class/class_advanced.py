"""
    之前的建议是使用将属性以单下划线开头，通过这种方式来暗示属性是受保护的，不建议外界直接访问，那么如果想访问可以通过属性的getter(访问器) 和setter(修改器)方法进行对应的操作。
"""


class Person(object):

    # 限定Person对象只能绑定_name, _age和_gender属性
    __slots__ = ('_name', '_age', '_gender')

    def __init__(self, name, age):
        self._name = name
        self._age = age

    # 访问器 - getter 方法
    @property
    def name(self):
        return self._name

    # 访问器 - getter 方法
    @property
    def age(self):
        return self._age

    # 修改器 - setter方法
    @age.setter
    def age(self, age):
        self._age = age

    def play(self):
        if self._age <= 16:
            print('%s 正在玩飞行棋.' % self._name)
        else:
            print('%s 正在玩斗地主.' % self._name)


from time import time, localtime, sleep


class Clock(object):
    def __init__(self, hour = 0, minute = 0, second = 0):
        self._hour = hour
        self._minute = minute
        self._second = second

    # 类方法需要用修饰器@classmethod来标识，告诉解释器这是一个类方法
    @classmethod
    def now(cls):
        ctime = localtime(time())
        return cls(ctime.tm_hour, ctime.tm_min, ctime.tm_sec)

    def run(self):
        """走字"""
        self._second += 1
        if self._second == 60:
            self._second = 0
            self._minute += 1
            if self._minute == 60:
                self._minute = 0
                self._hour += 1
                if self._hour == 24:
                    self._hour = 0

    def show(self):
        """显示时间"""
        return '%02d:%02d:%02d' % \
               (self._hour, self._minute, self._second)


def main():
    person = Person('边俊亭', 12)
    person.play()
    person.age = 22
    person.play()
    # person.name = '白元芳' # AttributeError: can't set attribute
    person._gender = '男'
    # AttributeError: 'Person' object has no attribute '_is_gay'
    # person._is_gay = True

    # 通过类方法创建对象并且获取系统时间
    clock = Clock.now()
    while True:
        print(clock.show())
        sleep(1)
        clock.run()


if __name__ == '__main__':
    main()

