# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:10:40 2024

@author: Zaria Ferté
"""

import math
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

tau = None
x1, x2, y1, y2 = 0, 0, 0, 0
rect_selector = None
phi_entry, m_entry = None, None
callback = None
omega=0

def get_phi_from_tau(tau):
    global omega
    print(omega)
    return math.degrees(math.atan(omega * tau))

def get_m_from_tau(tau):
    global omega
    return 1 / (math.sqrt(1 + (omega * tau) ** 2))

def calibration():
    global x1, x2, y1, y2, coef_phi, coef_m, phi_entry, m_entry
    tau = float(lifetime.get())*10**-6
    print(tau)
    phi_theo = get_phi_from_tau(tau)
    m_theo = get_m_from_tau(tau)
    selected_phi = phi_entry[y1:y2, x1:x2]
    selected_m = m_entry[y1:y2, x1:x2]
    phi_mean = np.mean(selected_phi)
    m_mean = np.mean(selected_m)
    print(phi_mean, m_mean)
    print(phi_theo, m_theo)
    coef_phi = phi_theo / phi_mean
    coef_m = m_theo / m_mean
    if callback:
        callback(coef_phi, coef_m)

def add_rectangle_selector(ax, image, fig):
    global rect_selector

    def onselect(eclick, erelease):
        global x1, x2, y1, y2, top
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        print(f"Selected area : x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        

    rect_selector = RectangleSelector(ax, onselect, useblit=True,
                                      button=[1],  # left click
                                      spancoords='pixels', interactive=True)
    fig.canvas.mpl_connect('button_press_event', rect_selector)
    fig.canvas.mpl_connect('button_release_event', rect_selector)

def display_image(image, title, top):
    global calibrate_button, lifetime
    global x1, x2, y1, y2
    if lifetime.get() != '':
        fig = Figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        canvas_fig = FigureCanvasTkAgg(fig, master=top)
        canvas_fig.draw()
        canvas_widget = canvas_fig.get_tk_widget() 
        canvas_widget.pack(pady=50, fill=tk.BOTH, expand=True)
        add_rectangle_selector(ax, image,fig)
        calibrate_button.pack_forget()
        print(x1,x2, y1, y2)
        ok_button = tk.Button(top, text="OK", command=lambda: stop_selection(top))
        ok_button.pack(padx=0, pady=0, side=tk.BOTTOM, expand=True)

def stop_selection(top):
    global rect_selector, indicator, x1, x2, y1, y2
    if ((x1, x2, y1, y2) != (0, 0, 0, 0)):
        if rect_selector is not None:
            rect_selector.set_active(False)
        for widget in top.pack_slaves():
            if isinstance(widget, FigureCanvasTkAgg):
                widget.get_tk_widget().pack_forget()
            elif isinstance(widget, tk.Button) and widget.cget("text") == "OK":
                widget.pack_forget()
        calibration()
        messagebox.showinfo("Lifetime calibration", "Calibration done, you can now launch your acquisition")
        top.destroy()
        
   

def start_calibration(root,ni, phi, m, cb, omega1):
    global lifetime, phi_entry, m_entry, callback, calibrate_button, omega
    omega = omega1
    phi_entry = phi
    m_entry = m
    callback = cb
    top = tk.Toplevel(root)
    top.title("Calibration")
    top.grab_set()

    tk.Label(top, text="Lifetime of the known component in µs :").pack()
    lifetime = tk.Entry(top)
    lifetime.pack()

    calibrate_button = tk.Button(top, text="Calibrate", command=lambda: display_image(ni, "Select an area where there is only the component (no background)", top))
    calibrate_button.pack()

    root.wait_window(top)
