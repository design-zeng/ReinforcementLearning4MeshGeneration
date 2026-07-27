from tkinter import *
import math
import json
from tkinter import filedialog


class MeshFrame(Frame):
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.points = []
        self.done = False  # Flag signalling we're done
        self.current = (0, 0)
        self.window_name = "Mesh Generation UI"  # Name for our window
        self.fore_color = '#502c69'
        self.back_color = '#000000'
        self.canvas = None
        self.last_draw = None
        self.area = None
        self.initUI()

    def initUI(self):
        self.master.title(self.window_name)
        self.pack(fill=BOTH, expand=True)

        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, pad=7)
        self.columnconfigure(3, pad=7)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(5, pad=7)

        # lbl = Label(self, text="Windows")
        # lbl.grid(sticky=W, pady=4, padx=5)

        self.cre_canvas()

        abtn = Button(self, text="Set density")
        abtn.grid(row=0, column=2)
        abtn.bind("<Button-1>", self.density)

        ccbtn = Button(self, text="Clear")
        ccbtn.grid(row=0, column=3, padx=4)
        ccbtn.bind("<Button-1>", self.clear_btn)

        ebtn = Button(self, text="Export")
        ebtn.grid(row=1, column=2, pady=4)
        ebtn.bind("<Button-1>", self.file_save)

        lbtn = Button(self, text="Import")
        lbtn.grid(row=1, column=3, pady=4)
        lbtn.bind("<Button-1>", self.load_file)

        area = Text(self, width=100, height=10)
        area.grid(row=4, column=0, columnspan=1, rowspan=1,
                  padx=0)
        self.area = area

        # hbtn = Button(self, text="Help")
        # hbtn.grid(row=5, column=0, padx=5)
        #
        # obtn = Button(self, text="OK")
        # obtn.grid(row=5, column=2)

    def load_file(self, event):
        filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                     filetypes=(("json files", "*.json"), ("all files", "*.*")))
        if not filename:
            return
        with open(filename, 'r') as fr:
            res = json.loads(fr.readline())

        self.clear_btn(None)
        self.points = [(p[0], p[1]) for p in res]

        self.canvas.delete("all")
        [self._create_circle(p[0], p[1], 3, fill="#000", outline="") for p in self.points]
        [self.canvas.create_text(p[0] - 2, p[1] - 2, fill="darkblue", font="Times 11 italic bold",
                                 text=f"{i}")
         for i, p in enumerate(self.points)]
        self.canvas.create_polygon(self.points, outline='#6e6565', fill="",
                                   width=2)
        self.done = True

    def file_save(self, event):
        f = filedialog.asksaveasfile(mode='w', defaultextension=".json")
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return

        # centriod = cal_centriod(self.points)
        # c_angle = clockwise_angle((self.points[0][0] - centriod[0], self.points[0][1] - centriod[1]),
        #                                (self.points[1][0] - centriod[0], self.points[1][1] - centriod[1]))

        if not self.check_clockwise():
        # if c_angle < math.pi:
        #     pass
        # else:
            self.points = list(reversed(self.points))

        text2save = json.dumps(self.points)  # starts from `1.0`, not `0.0`
        f.write(text2save)
        f.close()  # `()` was missing.

    def clear_btn(self, event):
        self.area.delete('1.0', END)
        self.canvas.delete("all")
        self.done = False
        self.points = []
        self.last_draw = None

    def density(self, event):
        # d = Tk()
        self.newWindow = Toplevel(self.master)
        den = Density(self.newWindow, self.points, self)

    def mouse_move(self, event):
        if self.done:  # Nothing more to do
            return

        if not len(self.points):
            return

        self.current = (event.x, event.y)
        try:
            self.canvas.delete(self.last_draw)
        except Exception as e:
            pass
        self.last_draw = self.canvas.create_line(self.points[-1][0], self.points[-1][1], event.x, event.y,
                           fill=self.fore_color)

    def l_mouse_down(self, event):

        if self.done:  # Nothing more to do
            return
        print("Adding point #%d with position(%d,%d)" % (len(self.points), event.x, event.y))
        self.points.append((event.x, event.y))
        self._create_circle(event.x, event.y, 3, fill="#000", outline="")

        self.last_draw = self.canvas.create_line(self.points[-1][0], self.points[-1][1], event.x, event.y,
                                                 fill=self.fore_color)
        self.area.insert(END, "Adding point #%d with position(%d,%d)" % (len(self.points), event.x, event.y))
        self.area.insert(END, "\n")

    def r_mouse_down(self, event):

        if self.done:  # Nothing more to do
            return
        print("Completing polygon with %d points." % len(self.points))

        self.canvas.delete("all")
        [self._create_circle(p[0], p[1], 3, fill="#000", outline="") for p in self.points]
        [self.canvas.create_text(p[0]-2, p[1]-2, fill="darkblue",font="Times 11 italic bold", text=f"{i}: ({p[0]}, {p[1]})")
         for i, p in enumerate(self.points)]
        self.canvas.create_polygon(self.points, outline='#6e6565', fill="",
                              width=2)
        self.done = True

    def _create_circle(self, x, y, r, **kwargs):
        return self.canvas.create_oval(x - r, y - r, x + r, y + r, **kwargs)

    def cre_canvas(self):
        canvas = Canvas(self, bg='white', width=800, height=600)
        canvas.grid(row=0, column=0, columnspan=2, rowspan=4,
                  padx=5)
        canvas.bind("<Motion>", self.mouse_move)
        canvas.bind("<Button-1>", self.l_mouse_down)
        canvas.bind("<Button-3>", self.r_mouse_down)
        self.canvas = canvas

    def check_clockwise(self):
        if len(self.points):
            clockwise = False
            if (sum([self.points[i - 1][0] * p[1] - self.points[i - 1][1] * p[0]
                     for i, p in enumerate(self.points)])) < 0:
                clockwise = not clockwise
            return clockwise


def cal_centriod(points):
    if not len(points):
        return
    points_len = len(points)
    centriod_x = sum([p[0] for p in points]) / points_len
    centriod_y = sum([p[1] for p in points]) / points_len
    return centriod_x, centriod_y


def clockwise_angle(A, B):
    AB = (B[0] - A[0], B[1] - A[1])
    # unit = (1, 0)

    theta = - math.atan2(- AB[1], AB[0])

    return theta if math.copysign(1, theta) >= 0 else 2 * math.pi + theta


class Density(Frame):
    def __init__(self, master, points, base_frame):
        super().__init__()
        self.master = master
        self.window_name = "Density setting"
        self.points = points
        self.base_frame = base_frame
        self.base_entry = None
        self.density_entries = []
        self.initUI()

    def initUI(self):
        self.master.title(self.window_name)
        self.pack(fill=BOTH, expand=True)

        # scrollbar = Scrollbar(self.master)
        # scrollbar.pack(side=RIGHT, fill=Y)
        #
        # canvas = Canvas(self.master, bg='white', width=100, height=400)
        # canvas.grid(row=0, column=0, columnspan=2, rowspan=4,
        #             padx=5)

        abtn = Button(self.master, text="Apply")
        abtn.grid(row=0, column=1, columnspan=2, sticky='E')
        abtn.bind("<Button-1>", self.calculate_density)

        max_length = self.calculate_max_base_length()
        Label(self.master, text=f"Base length (< {max_length})").grid(row=1)
        e1 = Entry(self.master)
        e1.insert(END, '1')
        e1.grid(row=1, column=1)
        self.base_entry = e1

        count = 0
        for i, p in enumerate(self.points):
            Label(self.master, text=f"Point {i}").grid(row=i + 2)
            e1 = Entry(self.master)
            e1.insert(END, '1')
            e1.grid(row=i + 2, column=1)
            self.density_entries.append(e1)
            count += 1


    def calculate_max_base_length(self):
        max_length = 0
        for i in range(len(self.points)):
            dist = self.distance(self.points[i - 1], self.points[i])
            if dist > max_length:
                max_length = dist
        return round(max_length, 2)

    def distance(self, point1, point2):
        dist = math.sqrt((point1[0] - point2[0]) ** 2 +
                         (point1[1] - point2[1]) ** 2)
        return dist

    def calculate_density(self, event):
        base_length = float(self.base_entry.get())
        points_density = {p: float(self.density_entries[i].get()) for i, p in enumerate(self.points)}
        print(points_density)
        list_points_density = [(k, v) for k, v in points_density.items()]
        res_points = []
        for i, (k, v) in enumerate(list_points_density):
            B = v * base_length
            A = list_points_density[i - 1][1] * base_length
            L = self.distance(list_points_density[i - 1][0], k)
            x = round((2 * L - A - B) / (A + B))
            e = (B - A) / x
            angle = clockwise_angle(list_points_density[i - 1][0], k)
            interpolations = [A * (i + 1) + e * (i ** 2 + i) / 2 for i in range(x)]
            interpolations.append(L)
            if i == len(list_points_density) - 1:
                if (len(res_points) + len(interpolations)) % 2 == 1:
                    interpolations.pop(int(len(interpolations) / 2))
            res_points.extend([(list_points_density[i - 1][0][0] + inter * math.cos(angle),
                                list_points_density[i - 1][0][1] + inter * math.sin(angle))
                               for inter in interpolations])

        # print(res_points)
        [self.base_frame._create_circle(p[0], p[1], 3, fill="#f00", outline="") for p in res_points]
        self.base_frame.points = res_points


if __name__=="__main__":
    root = Tk()
    # root.geometry("900x700+30+30")
    app = MeshFrame(root)
    root.mainloop()