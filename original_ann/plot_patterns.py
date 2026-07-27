import matplotlib.pyplot as plt
from general import data
import numpy as np
import turtle
import time
import math as m


def plot_patterns():
    inputs, output_types, outputs = data.get_patterns("pattern.txt")
    # transfered_data = data.data_transformation(np.concatenate((inputs, outputs), axis=1), 4, 5, 6, 7)

    points = inputs
    for i, point in enumerate(points):
        if i < 100:
            continue
        # plot existing points
        input_x = [point[i] for i in range(len(point)) if i % 2 == 0]
        input_y = [point[i] for i in range(len(point)) if i % 2 == 1]
        # input_x.append(input_x[0])
        # input_y.append(input_y[0])
        plt.plot(input_x, input_y, 'r-')

        #plot generated point and lines
        new_x = [point[2], outputs[i][0], point[6]]
        new_y = [point[3], outputs[i][1], point[7]]

        plt.plot(new_x, new_y, 'b-')

        plt.show()
        plt.close()

# plot_patterns()
def plot_flat_points(point):
    input_x = [point[i] for i in range(len(point)) if i % 2 == 0]
    input_y = [point[i] for i in range(len(point)) if i % 2 == 1]
    # input_x.append(input_x[0])
    # input_y.append(input_y[0])
    plt.plot(input_x, input_y, 'ro-')
    plt.show()


class Sheep(object):

    def __init__(self,xsize):
        self.t = turtle.Turtle()
        self.xsize = xsize
        t = self.t
        # 画笔设置
        t.screen.screensize(canvwidth=1000,canvheight=500,bg='white')
        t.pensize(2)
        t.speed(10)
        # t.hideturtle()
        #初始化画笔位置
        t.penup()
        t.setpos(self.xsize,0)
        t.pendown()

    # 设置画笔坐标
    def setxy(self,x,y):
        t = self.t
        t.penup()
        pos_x = t.position()[0]
        pos_y = t.position()[1]
        t.setpos(pos_x + x,pos_y + y)
        t.pendown()

    def create_sheep(self):
        t = self.t
        # 羊头
        self.setxy(-200,0)
        t.fillcolor('black')
        t.begin_fill()
        t.circle(100)
        t.end_fill()

        # 眼睛
        # 眼白
        print(t.position())
        self.setxy(-20,120)

        t.fillcolor('white')
        t.begin_fill()
        t.seth(45)
        t.circle(18,-280)
        t.seth(45)
        t.circle(-20,292)
        t.end_fill()
        # 眼珠
        self.setxy(3,12)
        t.fillcolor('black')
        t.begin_fill()
        t.seth(85)
        t.circle(10)
        t.seth(85)
        t.circle(-10)
        t.end_fill()
        # 眼心
        t.fillcolor('white')
        t.begin_fill()
        t.seth(85)
        t.circle(3)
        t.seth(85)
        t.circle(-3)
        t.end_fill()

        # 嘴
        self.setxy(0,-100)
        t.color('red')
        t.seth(300)
        t.forward(8)
        self.setxy(-1, 3)
        t.seth(0)
        t.circle(80,60)
        self.setxy(2, -2)
        t.seth(145)
        t.forward(8)
        t.color('black')

        # 耳朵
        self.setxy(-145,120)
        p1 = t.position()
        t.fillcolor('black')
        t.begin_fill()
        t.seth(0)
        t.circle(-120,20)
        p2 = t.position()
        t.setpos(p1)
        t.seth(60)
        t.circle(-30,120)
        t.goto(p2)
        t.end_fill()

        # 身体
        self.setxy(41,12)
        t.seth(45)
        t.circle(-150,100)
        t.pensize(5)
        t.seth(0)
        t.circle(-120,30)
        t.seth(60)
        t.circle(-15,320)
        t.seth(330)
        t.circle(-80,180)
        t.seth(210)
        t.circle(-80,90)

        #4条腿
        t.pensize(2)
        for leg in range(4):
            self.setxy(8+15*leg,0)
            t.seth(270)
            t.forward(80)
            t.seth(0)
            t.forward(8)
            t.seth(90)
            t.forward(80)

        #草
        self.setxy(-200,-80)
        p3 = t.position()
        t.color('green')
        t.fillcolor('green')
        t.begin_fill()
        t.seth(120)
        t.forward(30)
        t.seth(330)
        t.forward(30)
        t.seth(60)
        t.forward(40)
        t.seth(260)
        t.forward(45)
        t.setpos(p3)
        t.end_fill()


class Star():
    def __init__(self):
        self.t = turtle.Turtle()

    def create(self):
        self.t.fillcolor('red')  # 设置填充颜色为红色

        self.t.hideturtle()  # 隐藏箭头显示

        self.t.begin_fill()  # 开始填充

        while True:

            self.t.forward(200)

            self.t.right(144)

            if abs(self.t.pos()) < 1:
                break

        self.t.end_fill()  # 结束填充


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# import matplotlib.pyplot as plt
# import numpy as np
#
# # Fixing random state for reproducibility
# np.random.seed(19680801)
#
#
# def randrange(n, vmin, vmax):
#     '''
#     Helper function to make an array of random numbers having shape (n, )
#     with each number distributed Uniform(vmin, vmax).
#     '''
#     return (vmax - vmin)*np.random.rand(n) + vmin
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# n = 100
#
# # For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, marker=m)
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()

# if __name__ == '__main__':
#     # for x in (0,350):
#     #     sheep = Sheep(x)
#     #     sheep.create_sheep()
#     # time.sleep(5)
#     star = Star()
#     star.create()