# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:10:40 2024

@author: Zaria Ferté
"""


'''


Create and manage the fitting window, selection of a model define in the file 'fitt_model.py', acquisition of your points
Display of the fitting curve and you points and the best parameters for your equation


'''


import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from lmfit import Parameters
import pandas as pd
import pco
import fitt_model 

x1, x2, y1, y2 = 0, 0, 0, 0
previous_img=None
x,y=np.array([]),np.array([])
frame=None
fitted_parameters, fitted_function,fitted_name=None, None, None           
nb_points=0
omega=0
list_label=[]

def add_rectangle_selector(ax, image, fig):
    global x1, x2, y1, y2
    def onselect(eclick, erelease):
        global x1, x2, y1, y2, top
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)  
        
    rect_selector = RectangleSelector(ax, onselect, useblit=True,
                                      button=[1],  # left click
                                      spancoords='pixels', interactive=True)
    fig.canvas.mpl_connect('button_press_event', rect_selector)
    fig.canvas.mpl_connect('button_release_event', rect_selector)
    # fig.canvas.draw_idle()
 
        
def display_image(top, ni):
    global previous_img
      
    if previous_img:
        previous_img.pack_forget()
    image=ni
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(image, cmap='gray')
    
    
    ax.set_title("Select an area whithout background")
    ax.axis('off')
    canvas_fig = FigureCanvasTkAgg(fig, master=top)
    canvas_fig.draw()
    canvas_widget = canvas_fig.get_tk_widget() 
    canvas_widget.pack(pady=10)
    previous_img=canvas_widget

    add_rectangle_selector(ax, image,fig)
    
        
def update_param_fields(option, top, image_nb,camera, fc, event=None):
    global frame, nb_points, y,list_label
    y=np.array([])
    k=fitt_model.models_names.index(option.get())
    fitting_model=fitt_model.models_formula[k]  
    info=fitt_model.models_infos[k]
    nb_points=0
    entries = {}
    fixed_vars = {}
    entry_vars = {}
    entries_tab = []
    buttons=[]
    list_label=[]
    if frame:
        frame.destroy()
    def fit_callback():
        global x,y,fitted_parameters, fitted_function, fitted_name, omega,list_label
        x=np.array([])
        
        def plot_results2(params):
            global x,y
            fig = Figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            ax.set_title("Fitting")
            ax.scatter(x, y)
            ax.axis('on')
            ax.grid(True)
            ax.set_xlabel(str(fitting_model.independent_vars))
            ax.set_ylabel(str(info[0]))
            
            num_points = len(x)
            
            independent_vars_data = {name: data for name, data in zip(fitting_model.independent_vars, [x])}

            result = fitting_model.fit(y, params, **independent_vars_data)

            # print(result.fit_report())
        
            x_fine = np.linspace(np.min(x), np.max(x), 500)
            y_fine = fitting_model.func(x_fine, **result.params.valuesdict())
        
            x_fit = np.linspace(np.min(x), np.max(x), 100)
            y_fit = fitting_model.func(x_fit, **result.params.valuesdict())
            ax.plot(x_fit, y_fit, label='Best Fit', color='red')
            ax.legend()
            
            params = result.params.valuesdict()
            
            formula_label = tk.Label(top, text=f"Parameters: {params}")
            formula_label.grid(row=6, column=1, columnspan=3, padx=10, pady=10, sticky='ew')
            fitted_parameters=params
            fitted_function=(info[2],info[0],info[3])
            fitted_name=fitting_model.name.split('(')[1].rstrip(')')
            
            canvas_fig = FigureCanvasTkAgg(fig, master=top)
            canvas_fig.draw()
            canvas_widget = canvas_fig.get_tk_widget()
            canvas_widget.grid(row=3, column=1, columnspan=3, padx=10, pady=5, sticky='ew')
            return fitted_parameters, fitted_function, fitted_name
        
        for i in range (len(y)):
            y[i]=float(list_label[i].get())
        if len(y)<len(fitting_model.param_names):
            messagebox.showinfo("Error", f"Please calculate at least {len(fitting_model.param_names)} values before clicking on Fit.")
            top.lift()
        else:    
            for i in range (len(y)):
                if entries_tab[i].get() == '' and y[i]!='':
                    messagebox.showinfo("Error", "Please indicate the values before clicking on Fit.")
                    top.lift()
                    break
                   
                else:
                    x=np.append(x,float(entries_tab[i].get()))
            params = fitting_model.make_params()        
            for param in fitting_model.param_names:
                value = float(entries[param].get())
                params[param].set(value=value)
                if entries[param + '_min'].get() != '':
                    params[param].min = float(entries[param + '_min'].get())
                if entries[param + '_max'].get() != '':
                    params[param].max = float(entries[param + '_max'].get())
                params[param].vary = not(fixed_vars[param].get())
                
            fitted_parameters, fitted_function, fitted_name = plot_results2(params)
            
        
    def calc_callback(iid, col_index, image_nb,camera, fc):
        global y,val, nb_points,list_label
        if camera:
            messagebox.showinfo("Info", "Place the component and turn on the modulation light.")
            
            def validate(top1,top,im):
                global y1,y2, x1,x2, val
                if ((x1, x2, y1, y2) != (0, 0, 0, 0)):
                    val=np.mean(im[y1:y2, x1:x2])
                    if val != np.NaN:
                        if info[0]=='tau (µs)':
                            val=10**6*np.tan(np.radians(val))/omega
                        else:
                            val=info[3](val)
                        top.lift()
                        top1.destroy()
            def main():
                def acquisition_cam(camera):
                    frames=[]
                    camera.record(image_nb, mode='sequence')
                    for i in range(image_nb):
                        frames.append(camera.image(image_index=i))
                    camera.stop()
                    return frames
                frames = acquisition_cam(camera)
                liste = []
                for i in range(image_nb):
                    liste.append(frames[i][0])
                phi, m, ni, phasor = fc.calculate(liste)
                phi = np.degrees(phi)*-1
                return ni, phi      
                    
            top1 = tk.Toplevel(top)
            top1.title("ROI selection")
            param_label = tk.Label(top1, text=f"Select an area where there is only the component")
            ni, phi=main()
            
            display_image(top1, ni)
            next_button = tk.Button(top1, text="Ok", command=lambda:validate(top1,top, phi))
            next_button.pack(padx=0, pady=0, side=tk.RIGHT, expand=True)    
            top.wait_window(top1)
            if val != np.NaN:
                buttons[int(iid)-2].destroy()
                y=np.append(y,float(val))
                label = ttk.Label(table_values, text=round(val,3),borderwidth=2, relief="solid", style='Top.TLabel')
                label = ttk.Entry(table_values, style='Top.TEntry')
                label.insert(0, round(val,2))
                list_label.append(label)
                bbox = tree.bbox(iid, col_index)
                if bbox:
                    label.place(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
                top.lift()
        else:
            messagebox.showinfo("Error", "No camera found.")
            top.lift()
        
    def add_val(iid, columns):
        global nb_points
        nb_points+=1
        iid=str(int(iid)+nb_points)
        tree.insert("", "end", iid=iid, values=("", ""))
        for col_index, col_name in enumerate(columns):
            if col_name == str(info[1]):
                    calc_button = ttk.Button(table_values, text="calc", style= 'Top.TButton', command=lambda iid=iid, col_index=col_index: calc_callback(iid,col_index,image_nb,camera, fc))
                    buttons.append(calc_button)
                    tree.update_idletasks()
                    bbox = tree.bbox(iid, col_index)
                    if bbox:
                        calc_button.place(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
            else:
                entry_var = tk.StringVar()
                entry_vars[iid] = entry_var
                entry = ttk.Entry(table_values, textvariable=entry_vars[iid],style='Top.TEntry')
                entries_tab.append(entry)
                tree.update_idletasks()
                bbox = tree.bbox(iid, col_index)
                if bbox:
                    entry.place(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])      
    
    param_frame = ttk.LabelFrame(top, text="Calibration Parameters", style='Top.TLabelframe')
    param_frame.grid(row=1, column=6, padx=10, pady=10, columnspan=2,rowspan=2, sticky='ew')
    i=0
    frame=param_frame
    ttk.Label(param_frame, text="value",style='Top.TLabel').grid(row=0, column=1, padx=25, pady=5, columnspan=1, sticky='ew')
    ttk.Label(param_frame, text="min",style='Top.TLabel').grid(row=0, column=2, padx=10, pady=5, columnspan=1, sticky='ew')
    ttk.Label(param_frame, text="max",style='Top.TLabel').grid(row=0, column=3, padx=10, pady=5, columnspan=1, sticky='ew')
    
    for param in fitting_model.param_names:
        
        ttk.Label(param_frame, text=param, style='Top.TLabel').grid(row=i+1, column=0, padx=5, pady=5, columnspan=1, sticky='ew')
        entry = ttk.Entry(param_frame, width=5, style='Top.TEntry')
        entry.insert(0,fitting_model.make_params()[param].value)  # Parameters([('Top', <Parameter 'Top', value=6, bounds=[-inf:inf]>), ('Bottom', <Parameter 'Bottom', value=0.5, bounds=[-inf:inf]>), ('V50', <Parameter 'V50', value=7.5, bounds=[-inf:inf]>), ('slope', <Parameter 'slope', value=1, bounds=[-inf:inf]>)])
        entry.grid(row=i+1, column=1, padx=25, pady=5, columnspan=1, sticky='ew')
        entries[param] = entry
        
        min_entry = ttk.Entry(param_frame, width=5, style='Top.TEntry')
        min_entry.grid(row=i+1, column=2, padx=5, pady=5, columnspan=1, sticky='ew')
        entries[param + '_min'] = min_entry
        
        max_entry = ttk.Entry(param_frame, width=5,style='Top.TEntry')
        max_entry.grid(row=i+1, column=3, padx=5, pady=5, columnspan=1, sticky='ew')
        entries[param + '_max'] = max_entry
        
        fixed_var = tk.BooleanVar(0)
        ttk.Checkbutton(param_frame, text="fixed",variable=fixed_var, style='Top.TCheckbutton').grid(row=i+1, column=4, padx=15, pady=5, columnspan=1, sticky='ew')
        fixed_vars[param] = fixed_var
        i+=1
        
    table_values = ttk.LabelFrame(top, text="Calibration Points", style='Top.TLabelframe')
    table_values.grid(row=3, column=6, padx=10, pady=10, columnspan=2, sticky='ew')
    
    
    columns = fitting_model.independent_vars+ [info[0]]    
    tree = ttk.Treeview(table_values, columns=columns,style='Top.Treeview', show="headings", selectmode="browse", padding=5)
    
    
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor='center')
    
    tree.pack(side="top", fill="both", expand=True)
    
    for j in range(len(fitting_model.param_names)):
        iid = str(j+1)
        nb_points=0
        add_val(iid,columns)
    nb_point=len(fitting_model.param_names)
    button_frame = tk.LabelFrame(table_values, relief='flat')
    button_frame.pack(side=tk.BOTTOM, fill='x')
    add_value = ttk.Button(button_frame, text="add point", style= 'Top.TButton', command=lambda iid=iid: add_val(iid, columns))
    add_value.pack()           
    fitt_button = ttk.Button(button_frame, text="Fit", style= 'Top.TButton', command=lambda iid=iid: fit_callback())
    fitt_button.pack(fill='x')
    

            
def create_fitting(root, camera_entry, fc_entry, image_nb, omega_entry):
    global x,y,fitted_parameters, fitted_function,fitted_name, omega
    omega=omega_entry
    x,y=np.array([]),np.array([])
    top = tk.Toplevel(root)
    top.title("Fitting")
    ttk.Label(top, text="Select Model:", style='Top.TLabel').grid(row=1, column=1, columnspan=2, padx=10, pady=5, sticky='ew')
    selected_model = tk.StringVar()
    selected_model.set('')
    option = ttk.Combobox(top, textvariable=selected_model,style='Top.TCombobox', values=fitt_model.models_names, state='readonly')
    option.grid(row=1, column=3, columnspan=2, padx=10, pady=10, sticky='ew')
    option.bind('<<ComboboxSelected>>', lambda event: update_param_fields(option,top, image_nb,camera_entry, fc_entry, event))
    
    
    root.wait_window(top)
    if fitted_parameters and fitted_function and fitted_name:
        return fitted_parameters, fitted_function, fitted_name
    else:
        return (1,1,1)
 
