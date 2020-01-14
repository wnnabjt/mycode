class Student(object):
    # __init__是一个特殊方法用于在创建对象时进行初始化操作
    # 通过这个方法我们可以为学生对象绑定name 和 age两个属性
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def study(self, course_name):
        print('%s 正在学习%s.' % (self.name, course_name))

    # PEP 8要求标识符的名字用全小写多个单词用下划线链接
    # 但是部分程序员和公司更倾向于使用驼峰命名法
    def watch_movie(self):
        if self.age < 18:
            print('%s 只能看 《熊出没》.' % self.name)
        else:
            print('%s 正在观看岛国爱情大电影.' % self.name)

class Test:

    def __init__(self, foo):
        self.__foo = foo
    # 在给属性命名时可以用两个下划线作为开头，表示希望属性是私有的。 但是实际上要遵循单下划线开头表示属性受保护，由于实际开发中不建议将属性设置为私有，这会导致子类无法访问。
    def __bar(self):
        print(self.__foo)
        print('__bar')


# 封装:隐藏一切可以隐藏的实现细节，只向外界暴露（提供）简单的编程接口

from time import sleep

class Clock(object):
    #  TODO:数字时钟
    def __init__(self, hour = 0, minute = 0, second = 0):
        # 初始化方法
        self._hour = hour
        self._minute = minute
        self._second = second

    def run(self):
        # 走字
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
        # 显示时间
        return '%02d:%02d:%02d' % (self._hour, self._minute, self._second)

from math import sqrt

class Point(object):
    def __init__(self, x = 0, y = 0):
        # 初始化方法
        self.x = x
        self.y = y

    def move_to(self, x, y):
        # 移动到指定位置
        self.x = x
        self.y = y

    def move_by(self, dx, dy):
        # 移动指定的增量
        self.x += dx
        self.y += dy

    def distance_to(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return sqrt(dx ** 2 + dy ** 2)

    def __str__(self):
        return '(%s, %s)' % (str(self.x), str(self.y))



def main():
    """
    stu1 = Student('边俊亭', 21)

    stu1.study('Python程序设计')
    stu1.watch_movie()
    stu2 = Student('王大锤', 15)
    stu2.study('C语言')
    stu2.watch_movie()
    clock = Clock(23, 59, 58)
    while True:
        print(clock.show())
        sleep(1)
        clock.run()
    """
    p1 = Point(3, 5)
    p2 = Point()
    print(p1)
    print(p2)
    p2.move_by(-1, 2)
    print(p2)
    print(p1.distance_to(p2))

if __name__ == '__main__':
    main()




