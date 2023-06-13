from functions import *
from matplotlib import use

use('Agg')
use('TkAgg')
import matplotlib.pyplot as plt
import pylidc as pl
import os
import tkinter
from tkinter import ttk
from tkinter.messagebox import showinfo
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector


class Layer_selector(tkinter.Frame):
    def __init__(self, ig, golds, images_list, master):
        super().__init__(master)
        self.stats = None
        self.master = master
        self.place()

        self.frameSecundario = tkinter.Frame(
            self.master)  # Armazena os switchs para mostrar a  imagem binaria e ligar a selecao do ROI
        self.frameTerciario = tkinter.Frame(
            self.master)  # Armazena a entrada do id do paciente que sera analsiado (LIDC-IDRI-XXXX)

        self.layer = 0
        self.fig = None
        self.ax = None

        self.img_oculta = None

        self.img_class = np.ones((512, 512, 50), float)
        self.img_class[0, 0, 0] = 0
        self.img_size = self.img_class.shape

        self.gs_class = None

        # self.img_class = ig
        # self.img_size = ig.shape
        #
        # self.gs_class = golds
        self.img_sgmentada = None

        self.ROI = None
        self.switch_variable = tkinter.BooleanVar(self.master, value=True)
        self.img_switch_variable = tkinter.BooleanVar(self.master, value=True)

        self.slider = tkinter.Scale(self.master, command=self._slide, from_=0, to=self.img_class.shape[2] - 1,
                                    orient=tkinter.HORIZONTAL, length=399)
        self.slider.pack(side=tkinter.BOTTOM)

        self.switch = tkinter.Checkbutton(self.frameSecundario, text="Select ROI", variable=self.switch_variable,
                                          indicatoron=False, onvalue=False, offvalue=True, width=8,
                                          command=self.toggle_selector)

        self.switch.pack(side=tkinter.LEFT)

        espaco = tkinter.Label(self.frameSecundario, text=" ", width=5)
        espaco.pack(side=tkinter.LEFT)

        self.switch_gs = tkinter.Checkbutton(self.frameSecundario, text="Show Bin", variable=self.img_switch_variable,
                                             indicatoron=False, onvalue=False, offvalue=True, width=8,
                                             command=self.switch_img)

        self.switch_gs.pack(side=tkinter.LEFT)

        self.frameSecundario.pack(side=tkinter.BOTTOM)

        self.stats_buttom = tkinter.Button(self.master, text="Stats", command=self.popup_avalicao)
        self.stats_buttom.pack(side=tkinter.BOTTOM, expand=True, fill="x")

        self.quit_buttom = tkinter.Button(self.master, text="Run", command=self.run_method)
        self.quit_buttom.pack(side=tkinter.BOTTOM, expand=True, fill="x")

        self.im = None
        self.canvas = None
        self.create_canvas()

        # Frame terciario contendo o combobox com a lista de imagens disponiveis para serem lidas ----------------------

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frameTerciario, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tkinter.LEFT)

        self.paciente_input = ttk.Combobox(self.frameTerciario, state="readonly", values=images_list)
        self.buttom_read_img_selected = tkinter.Button(self.frameTerciario, text="Search", command=self.read_pacient)

        self.paciente_input.pack(side=tkinter.RIGHT)
        self.buttom_read_img_selected.pack(side=tkinter.RIGHT)

        self.frameTerciario.pack(side=tkinter.TOP, fill=tkinter.X, expand=True)
        # --------------------------------------------------------------------------------------------------------------

        self.selector = RectangleSelector(self.ax, self.select_callback,
                                          useblit=True,
                                          button=[1, 3],  # disable middle button
                                          minspanx=5, minspany=5,
                                          spancoords='pixels',
                                          interactive=True,
                                          state_modifier_keys={'clear': 'escape'})
        self.selector.set_active(False)

        self.slider.set(self.layer)

        # self.close_window()

    def read_pacient(self):
        selection = self.paciente_input.get()
        img1, gs1, _ = create_GoldSTD(selection)

        img1 = img1 / img1.max()
        self.img_class = img1.astype(float)

        gs1 = gs1 / gs1.max()
        self.gs_class = gs1.astype(float)

        self.img_size = self.img_class.shape

        self.ROI = None
        self.slider.configure(to=self.img_size[2] - 1)

    def select_callback(self, eclick, erelease):
        if eclick.button == 1:
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            self.ROI = (x1, x2, y1, y2)
            # print(self.ROI)

    def create_canvas(self):
        if self.img_class is not None:
            self.fig, self.ax = plt.subplots(1, 1)
            plt.axis('off')

            self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
            self.im = self.ax.imshow(self.img_class[:, :, int(self.layer)], cmap='gray')
            plt.close(self.fig)

            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tkinter.BOTTOM, expand=True, fill=tkinter.BOTH)

    def _slide(self, value):
        if self.im is not None:
            self.im.set_data(self.img_class[:, :, self.slider.get()])
            self.canvas.draw()
            self.layer = self.slider.get()

    def toggle_selector(self):
        if self.switch_variable.get():
            self.selector.set_active(False)
        else:
            self.selector.set_active(True)

    def run_method(self):
        if self.ROI:
            self.img_sgmentada, self.gs_class = fuzzy_adpatado(self.img_class, self.gs_class, self.ROI, self.layer)
            self.stats = avaliarSegmentacao(self.img_sgmentada, self.gs_class)

            showinfo(title=None, message="Processo Finalizado!")

    def switch_img(self):
        if self.img_sgmentada is not None:
            if not self.img_switch_variable.get():
                print("Swithc Img on")
                self.img_oculta = self.img_class
                self.img_class = self.img_sgmentada * self.img_class + self.gs_class * self.img_class
            else:
                print("Swithc Img off")
                # self.img_oculta = self.img_class
                self.img_class = self.img_oculta

    def popup_avalicao(self):
        if self.stats:
            win = tkinter.Toplevel()
            win.wm_title("Window")
            # [vp, fp, fn, od, Or]
            var1 = tkinter.StringVar()
            var1.set(f"Verdadeiro positivo: {self.stats[0]}%")
            label1 = tkinter.Label(win, textvariable=var1, relief=tkinter.RAISED)
            label1.grid(row=0, column=0)

            var2 = tkinter.StringVar()
            var2.set(f"Falso positivo: {self.stats[1]}%")
            label2 = tkinter.Label(win, textvariable=var2, relief=tkinter.RAISED)
            label2.grid(row=1, column=0)

            var3 = tkinter.StringVar()
            var3.set(f"Falso negativo: {self.stats[2]}%")
            label3 = tkinter.Label(win, textvariable=var3, relief=tkinter.RAISED)
            label3.grid(row=2, column=0)

            var4 = tkinter.StringVar()
            var4.set(f"Overlap dice: {self.stats[3]}%")
            label4 = tkinter.Label(win, textvariable=var4, relief=tkinter.RAISED)
            label4.grid(row=3, column=0)

            var5 = tkinter.StringVar()
            var5.set(f"Overlap ratio: {self.stats[4]}%")
            label5 = tkinter.Label(win, textvariable=var5, relief=tkinter.RAISED)
            label5.grid(row=4, column=0)

            # var6 = tkinter.StringVar()
            # var6.set(f"Verdadeiro positivo: {self.stats[0] * 100}%")
            # label6 = tkinter.Label(win, textvariable=var5, relief=tkinter.RAISED)

            b = tkinter.Button(win, text="Quit", command=win.destroy)
            b.grid(row=5, column=0)

    def close_window(self):
        self.master.quit()
        self.master.destroy()


path_imgs = pl.query(pl.Scan).order_by(pl.Scan.patient_id).first().get_path_to_dicom_files().split("\\LIDC-IDRI")[0]
imgs_list = os.listdir(path_imgs)

img, gs, nod = create_GoldSTD(imgs_list[0])

root = tkinter.Tk()
root.geometry('550x700')
root.title('Volumetria de Tumor De Pulmao')
app = Layer_selector(ig=img.copy(), golds=gs.copy(), images_list=imgs_list, master=root)
# app = Layer_selector(images_list=imgs_list, master=root)
app.mainloop()
