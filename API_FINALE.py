# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:19:56 2024

@author: Zaria Ferté
"""

import warnings
# Ignores warning of type FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog, messagebox, PhotoImage, Label
from PIL import Image, ImageTk
import pco
import cv2
import numpy as np
import json
import pandas as pd
import pop_calibration as calibration
import importlib
import math
import fitz  # PyMuPDF
from matplotlib import patches
from matplotlib.widgets import RectangleSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
from scipy.signal import wiener
from lmfit import Parameters
from RangeSlider import RangeSlider
from tooltip import Tooltip
import fitting
from fitting import create_fitting
import dill

#  Global variables  #
image_labels=[]
(x1, x2, y1, y2)=(0,0,0,0)
nb_point=0
previous_img=None
x,y=[],[]
task=0
camera=None
phase_number= ''
image_nb = None
bg_corr = ''
tableau_donnees = [["Data", "Whole image", "ROI"],
        ["Mean", " ", " "],
        ["Min", " ", " "],
        ["Max", " ", " "],
        ["standard deviation", " ", " "]]

data=[[[],[]],[[],[]],[[],[]]]
headers=["Whole image", "ROI"]
index=["Intensity (mean, min, max)", "Phase", "Modulation"]
phase_number_to_int = {
            'manual shifting': 2,
            '2 phases': 2,
            '4 phases': 4,
            '8 phases': 8,
            '16 phases': 16}
canvas_list=[]
button_list=[]
i=0
j=0
rect_selectors = []
im_liste = []
phasor=None
bg_average= np.zeros((1008,1008))
omega = 0
coef_phi=1
coef_m=1
global_treshold=0.08
current_image=None 
fitted_parameters, fitted_function, fitted_name=[],[],[]
dist=None

#   SAVE THE PARAMETERS    #

def save_settings():
    global bg_average, coef_phi, coef_m,fitted_parameters, fitted_function, fitted_name
    settings = {
        "exposure": exposure_entry.get(),
        "frequency": freq.get(),
        # "source": source_entry.get(),
        "source": "intern",
        "waveform": form_entry.get(),
        "phases": nb_phase_entry.get(),
        "symmetry": symmetry_entry.get(),
        "phase_order": phase_order_entry.get(),
        "tap": tap_entry.get(),
        "background_correction": bg_entry.get(),
        "bg calibration": bg_average.tolist(),
        "coef phi": str(coef_phi),
        "coef m": str(coef_m)
        # "fitted parameters": fitted_parameters, 
        # "fitted function": fitted_function, 
        # "fitted name":fitted_name
    }
    data_to_save = {
    "fitted parameters": fitted_parameters,
    "fitted function": fitted_function,
    "fitted name": fitted_name}
    with open('settings.json', 'w') as f:
        json.dump(settings, f)
    with open('settings.pkl', 'wb') as fichier:
        dill.dump(data_to_save, fichier)    

def load_settings():
    global bg_average, coef_phi, coef_m,fitted_parameters, fitted_function, fitted_name
    try:
        with open('settings.json', 'r') as f: 
            settings = json.load(f)
        exposure_entry.insert(0, settings["exposure"])
        freq.insert(0, settings["frequency"])
        source_entry.set(settings["source"])
        form_entry.set(settings["waveform"])
        nb_phase_entry.set(settings["phases"])
        symmetry_entry.set(settings["symmetry"])
        phase_order_entry.set(settings["phase_order"])
        tap_entry.set(settings["tap"])
        bg_entry.set(settings["background_correction"])
        bg_average=np.array(settings["bg calibration"])
        coef_phi=float(settings["coef phi"])
        coef_m=float(settings["coef m"])
        # fitted_parameters=settings["fitted parameters"]: , 
        # fitted_function=settings["fitted function"], 
        # fitted_name="fitted name"
        with open('settings.pkl', 'rb') as fichier:
            settings = dill.load(fichier)
        
        fitted_parameters = settings["fitted parameters"]
        fitted_function = settings["fitted function"]
        fitted_name = settings["fitted name"]
    except FileNotFoundError:
        # print("Settings file not found. Using defaults.")
        None

#  When window is closed, root is detroy and parameters are saved  #
def on_closing():   
    save_settings()
    root.destroy()

def set_task(val):
    global task
    task=val
    
# To save each graphs  #
def save_plot(fig):
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if file_path:
        fig.savefig(file_path)

# Export the datas in excel  #
def export_tableau_to_excel(headers, index):
    global data
    df = pd.DataFrame(data, columns=headers, index = index)
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
    if file_path:
        df.to_excel(file_path, index=True)

#  Manage exceptions
def catch_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            #messagebox.showerror("Error", f"Something went wrong : Please, check if the camera is turned on and unused by another application")
            messagebox.showerror("Error", f"Something went wrong : {e}")
            # print(f"An error occured : {e}")
    return wrapper


#  Calculates how many images should be capture according to the parameters set  #
def get_image_nb(phase_number,phase_symmetry, phase_order,tap_select,asymmetry_correction):

        n = phase_number
        s = phase_symmetry
        o = phase_order
        t = tap_select
        c = asymmetry_correction

        if (n == 'manual shifting' and (t == 'tap A' or t== 'tap B')) or\
            (n == '2 phases' and s == 'singular' and (t == 'tap A' or t== 'tap B')):
            return 1
        
        elif (n == 'manual shifting' and t == 'both')or\
                (n == '2 phases' and s == 'singular' and t == 'both') or \
                    (n == '2 phases' and s == 'twice' and (t == 'tap A' or t== 'tap B')) or \
                        (n == '2 phases' and s == 'twice' and o == 'opposite' and t == 'both' and c == 'average') or\
                            (n == '4 phases' and s == 'singular' and (t == 'tap A' or t== 'tap B') and c == 'off'):
            return 2
        
        elif (n == '2 phases' and s == 'twice' and t == 'both') or\
                (n == '4 phases' and s == 'singular' and t == 'both' and c == 'off') or\
                    (n == '4 phases' and s == 'twice' and o == 'ascending' and (t == 'tap A' or t== 'tap B') and c == 'off') or\
                        (n == '4 phases' and s == 'twice' and o == 'opposite' and t == 'both' and c == 'average') or\
                            (n == '4 phases' and s == 'twice' and o == 'opposite' and t == (t == 'tap A' or t== 'tap B') and c == 'off') or\
                                (n == '8 phases' and s == 'singular' and (t == 'tap A' or t== 'tap B') and c == 'off'):
            return 4
    
        elif (n == '4 phases' and s == 'twice' and t == 'both' and c == 'off') or\
                (n == '8 phases' and s == 'singular' and t == 'both' and c == 'off') or\
                    (n == '8 phases' and s == 'twice' and o == 'ascending' and (t == 'tap A' or t== 'tap B') and c == 'off') or\
                        (n == '8 phases' and s == 'twice' and o == 'opposite' and c == 'average') or\
                            (n == '16 phases' and s == 'singular' and (t == 'tap A' or t== 'tap B') and c == 'off'):
            return 8
        
        elif (n == '8 phases' and s == 'twice' and t == 'both' and c == 'off') or\
                (n == '16 phases' and s == 'singular' and t == 'both' and c == 'off') or\
                    (n == '16 phases' and s == 'twice' and o == 'ascending' and (t == 'tap A' or t== 'tap B') and c == 'off') or\
                        (n == '16 phases' and s == 'twice' and o == 'opposite' and t == 'both' and c == 'average') or\
                            (n == '16 phases' and s == 'twice' and o == 'opposite' and (t == 'tap A' or t== 'tap B') and c == 'off'):
            return 16

        elif n == '16 phases' and s == 'twice' and t == 'both' and c == 'off':
            return 32

        else:
            return ('error')

#  Sets all the parameters (entered by user) and create an instance of camera  #
def initialisation_cam(camera):
    global image_nb, phase_number, nb_phase, bg_corr, omega
    try:
        # Récupérer les valeurs de l'utilisateur
        expos_time=float(exposure_entry.get())
        frequency = int(freq.get())
        # source_select=str(source_entry.get())
        source_select="intern"
        output_waveform=str(form_entry.get())
        phase_number=str(nb_phase_entry.get())
        phase_symmetry=str(symmetry_entry.get())
        phase_order=str(phase_order_entry.get())
        tap_select=str(tap_entry.get())
        bg_corr=str(bg_entry.get())
        phase_number= str(nb_phase_entry.get())
        nb_phase=phase_number_to_int[phase_number]
        
        asymmetry_correction = "off" #or average
        output_mode = "default"
        omega=2*math.pi*frequency    
        image_nb  = get_image_nb(phase_number,phase_symmetry, phase_order,tap_select,asymmetry_correction)
        if camera is not None:
        
            camera.exposure_time = expos_time
            camera.set_flim_configuration(
                frequency=frequency,
                phase_number=phase_number,
                source_select=source_select,
                output_waveform=output_waveform,
                phase_symmetry=phase_symmetry,
                phase_order=phase_order,
                tap_select=tap_select,
                asymmetry_correction=asymmetry_correction,
                output_mode=output_mode
                )
        flim=pco.Flim(phase_number, phase_symmetry, phase_order, tap_select, asymmetry_correction)
        frames=[]
        return frames, camera, flim
    except ValueError:
        # print("Error in the values entered")
        messagebox.showinfo("Error", "Error in the values entered")
    
#  Removes graphs from the interface and set each cell of the table to 'None' #
def clean():
    global canvas_list, button_list,rect_selectors,im_liste, tau_phi, tau_m
    global i
    i=0
    im_liste=[]
    rect_selectors=[]
    tau_phi.config(text="")
    tau_mod.config(text="")
    std.config(text="")
    for canvas in canvas_list:
        if canvas and canvas.get_tk_widget().winfo_exists():
            canvas.get_tk_widget().pack_forget()
    for button in button_list:
        if button:
            button.pack_forget()
    for i in range(4):
        tableau_donnees[i+1][1]='None'
        tableau_donnees[i+1][2]='None'
    write_tab(tableau_donnees)
            
    
#  Transforms a matlab plot to a tkinter graph and add the rectangle selector possibility  #  
def plt_to_tk(image, min_val, max_val, mean_val, title, _cmap, vmin=None, vmax=None):
    global i, task, canvas_list, button_list
    fig = Figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    if vmin is None:
        vmin = min_val
    if vmax is None:
        vmax = max_val

    im = ax.imshow(image, cmap=_cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')

    colorbar = fig.colorbar(im)
    canvas_fig = FigureCanvasTkAgg(fig, master=canvas)
    canvas_fig.draw()
    canvas_list.append(canvas_fig)
    canvas_widget = canvas_fig.get_tk_widget()
    canvas_widget.pack(pady=50, side=tk.LEFT, fill=tk.BOTH, expand=True)
    if task==6:
        add_rectangle_selector2(ax,image, title, fig)
    else:
        add_rectangle_selector(ax, image, title, fig)
    button_save_plot = ttk.Button(canvas, text=f"Save {title}", style="small3.TButton", command=lambda: save_plot(fig))
    button_save_plot.pack(side=tk.LEFT, padx=0)
    button_list.append(button_save_plot)
    i += 1

#  Manages the rectangle selector tool and update the table 'data' with the min, max, mean values  #
def add_rectangle_selector(ax, image,title,fig):
    global data,rect_selectors, im_liste, j, formulas, dist
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        for rect_selector in rect_selectors:
            rect_selector.set_active(True)
            rect_selector.extents = (x1, x2, y1, y2) 
            rect_selector.update()
        k=0
        if len(im_liste) ==4:
            for im in im_liste:
                if x1 == x2 and y1 == y2:
                    selected_mean=im[x1][y1]
                    selected_min,selected_max= selected_mean, selected_mean
                    dist_im=dist[x1][y1]
                else:
                    selected_area = im[y1:y2, x1:x2]
                    selected_min, selected_max, _, _ = cv2.minMaxLoc(selected_area)
                    selected_mean = np.mean(selected_area)
                    dist_im=dist[y1:y2, x1:x2]
                if k<3:
                    data[k][1]=selected_mean, selected_min,selected_max, np.std(im[y1:y2, x1:x2])
                k+=1
            tau_phi.config(text=f"{round(10**6*tau_from_phi(data[1][1][0]),4)}")
            tau_mod.config(text=f"{round(10**6*tau_from_mod(data[2][1][0]),4)}")    
            std.config(text=f"{round(np.std(dist_im),3)}")
            
        if title == 'Intensity':
            button_intensity()
        if title == 'Phase':
            button_phase()
        if title == 'Modulation':
            button_modulation()
            
    rect_selector = RectangleSelector(ax, onselect, useblit=True,
                                      button=[1],  # left click
                                      spancoords='pixels', interactive=True)
    rect_selectors.append(rect_selector)
    im_liste.append(image)
    
#  Removes the image displayed during live  #
def remove ():
    global image_labels
    for label in image_labels:
        label.destroy()
    image_labels=[]
    
#  Captures images  #
# @catch_exceptions
def acquisition_cam(camera, frames):
    global image_nb
    camera.record(image_nb, mode='sequence')
    for i in range(image_nb):
        frames.append(camera.image(image_index=i))
    camera.stop()
    return frames

#  Applies the Wiener filter and the MedianBlur filter on every images on the 'liste' and return the liste filtered  #
def two_filters(liste):
    i=0
    for image in liste:
        
        wiener_filtered = wiener(image, (5, 5))
        image = cv2.medianBlur(image, 3)
        liste[i]=image
        i+=1
    return liste
   
#  Applies a background correction by substracting the reference 'bg_average' (which correspond
#  to an average of the noise obtained during the calibration) to evry image on the liste and return liste  #
def bg_correction (liste):
    global bg_average
    threshold=0.8
    i=0
    for image in liste:
        if image is None:
            raise ValueError("There is no image in the list.")
        corrected_image=image-bg_average
        corrected_image=np.clip(corrected_image,0,16384)
        liste[i]= corrected_image.astype(image.dtype)
        i+=1
    return liste

def apply_mask_to_image(image, mask):
    """
    Apply a binary mask on a gray image.
    
    Parameters:
    - image : array-like, image en niveaux de gris.
    - mask : array-like, masque binaire (valeurs 0 et 1).
    
    Returns:
    - result : array-like
    """
    binary_mask = (mask > 0).astype(np.uint8)
    result = cv2.bitwise_and(image, image, mask=binary_mask)
    return result

#  Manages the mains functions
#  TASK:
#       =0 to stop live
#       =1 for the live record
#       =2 for the acquisition
#       =3 for the background calibration
#       =4 for calibrate a known lifetime
#       =5 for the fitting

@catch_exceptions
def Main():
    canvas.delete("all")
    remove()
    clean()
    global task,bg_corr, omega, current_image, bg_average, phasor, global_treshold,camera, dist
     
    with pco.Camera() as camera:
        frames, camera, fc =initialisation_cam(camera)
        label = Label(canvas)
        
        while task==1:
            try:
                camera.record(1,mode='sequence')
                image, metadata = camera.image()
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                image_pil = Image.fromarray(image)
                image_tk = ImageTk.PhotoImage(image_pil)
                label.config(image= image_tk)
                label.image = image_tk  # keep a reference for delete it
                label.pack()
                image_labels.append(label)
                root.update()
            except Exception as e:
                return(e)
        camera.stop()  # Stop direct acquisition
        cv2.destroyAllWindows()          
        
        if task==2:
            frames = acquisition_cam(camera, frames)
            liste = []
            for i in range(image_nb):
                liste.append(frames[i][0])
            if bg_corr== 'Yes':
                liste=bg_correction(liste)
            if apply_filters_var.get():
                liste=two_filters(liste)
            phi, m, ni, phasor = fc.calculate(liste)
            phi = np.degrees(phi)*-1
            
            if bg_corr== 'Yes':
                ret, mask = cv2.threshold(ni, global_treshold, 255, cv2.THRESH_BINARY) #adjust the treshold
                phi = apply_mask_to_image(phi, mask)
                m = apply_mask_to_image(m, mask)
                ni = apply_mask_to_image(ni, mask)
            
            current_image=[ni,coef_phi*phi,coef_m*m]
            min_val, max_val, _, _ = cv2.minMaxLoc(current_image[0])
            mean_val=cv2.mean(current_image[0])[0]
            slider_intensity = RangeSlider(right_frame, min_val, max_val, nb=2, width=400, height=100)
            slider_intensity.grid(row=51, column=0, columnspan=2, pady=10, padx=20, sticky='ew')
            slider_intensity.set_title('Intensity scale')
            min_val, max_val, _, _ = cv2.minMaxLoc(current_image[1])
            mean_val=cv2.mean(current_image[1])[0]
            slider_phase = RangeSlider(right_frame, min_val, max_val, nb=2, width=400, height=100)
            slider_phase.grid(row=52, column=0, columnspan=2, pady=10, padx=20, sticky='ew')
            slider_phase.set_title('Phase scale')
            min_val, max_val, _, _ = cv2.minMaxLoc(current_image[2])
            mean_val=cv2.mean(current_image[2])[0]
            slider_modulation = RangeSlider(right_frame, min_val, max_val, nb=2, width=400, height=100)
            slider_modulation.grid(row=53, column=0, columnspan=2, pady=10, padx=20, sticky='ew')
            slider_modulation.set_title('Modulation scale')
            dist=phi
            for i in range(len(phi)):
                for j in range (len(phi)):
                    dist[i][j]=10**6*tau_from_phi(phi[i][j])    
            min_val, max_val, _, _ = cv2.minMaxLoc(dist)
            mean_val=cv2.mean(dist)[0]
            slider_dist = RangeSlider(right_frame, min_val, max_val, nb=2, width=400, height=100)
            slider_dist.grid(row=53, column=0, columnspan=2, pady=10, padx=20, sticky='ew')
            slider_dist.set_title('Lifetime distrib scale')
            
            def update_display():
                global current_image, dist
                clean()
                vmin,vmax=slider_intensity.get_minmax()
                plt_to_tk(current_image[0], np.min(current_image[0]), np.max(current_image[0]), np.mean(current_image[0]), 'Intensity', 'gray', vmin=vmin, vmax=vmax)
                data[0][0]= [np.mean(current_image[0]),np.min(current_image[0]), np.max(current_image[0]), np.std(current_image[0])]
                vmin,vmax=slider_phase.get_minmax()
                plt_to_tk(current_image[1], np.min(current_image[1]), np.max(current_image[1]), np.mean(current_image[1]), 'Phase', 'rainbow', vmin=vmin, vmax=vmax)
                data[1][0]= [np.mean(current_image[1]),np.min(current_image[1]), np.max(current_image[1]), np.std(current_image[1])]
                vmin,vmax=slider_modulation.get_minmax()
                plt_to_tk(current_image[2], np.min(current_image[2]), np.max(current_image[2]), np.mean(current_image[2]), 'Modulation', 'rainbow', vmin=vmin, vmax=vmax)
                data[2][0]= [np.mean(current_image[2]),np.min(current_image[2]), np.max(current_image[2]), np.std(current_image[2])]
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist)
                vmin,vmax=slider_dist.get_minmax()
                mean_val=cv2.mean(dist)[0]
                plt_to_tk(dist, min_val, max_val, mean_val, 'Lifetime distribution', 'rainbow', vmin=vmin, vmax=vmax)
            update_display()
            slid_button = ttk.Button(right_frame, text="Apply changes", style="small2.TButton",command=update_display)
            slid_button.grid(row=56, column=0, columnspan=2, pady=20, padx=10, sticky='ew')
            
        if task==3:
            camera.record(150, mode= 'sequence')
            images,_=camera.images()
            bg_average=np.mean(images,axis=0)
            messagebox.showinfo("Background calibration", "Calibration done, you can now launch your acquisition")
        if task==4:
            frames = acquisition_cam(camera, frames)
            liste = []
            for i in range(image_nb):
                liste.append(frames[i][0])
            if bg_corr== 'Yes':
                liste=bg_correction(liste)
            if apply_filters_var.get():
                liste=two_filters(liste)
            phi, m, ni, phasor = fc.calculate(liste)
            phi=(np.degrees(phi))*-1
            camera.stop()
            calibration.start_calibration(root,ni,phi, m, receive_coefficients, omega)
        if task==5:
            a,b,c=fitting.create_fitting(root, camera,fc, image_nb, omega)
            if (a,b,c)!=(1,1,1):
                if c in fitted_name:
                    k=fitted_name.index(c)
                    fitted_parameters[k]=a
                    fitted_function[k]=b
                else:
                    fitted_parameters.append(a)
                    fitted_function.append(b)
                    fitted_name.append(c)
        if task==6:
            frames = acquisition_cam(camera, frames)
            liste = []
            for i in range(image_nb):
                liste.append(frames[i][0])
            if bg_corr== 'Yes':
                liste=bg_correction(liste)
            if apply_filters_var.get():
                liste=two_filters(liste)
            phi, m, ni, phasor = fc.calculate(liste)
            phi = np.degrees(phi)*-1
            
            if bg_corr== 'Yes':
                ret, mask = cv2.threshold(ni, global_treshold, 255, cv2.THRESH_BINARY) #adjust the treshold
                phi = apply_mask_to_image(phi, mask)
                m = apply_mask_to_image(m, mask)
                ni = apply_mask_to_image(ni, mask)
            current_image=[ni,coef_phi*phi,coef_m*m]
            
                
                
'''All API buttons + input parameters + button functions'''
def start_live():
    global task
    task=1
    clean()
    draw_button = ttk.Button(right_frame, text="Stop Live", command=stop_live)
    draw_button.grid(row=47, column=0, columnspan=2, pady=20, padx=10, sticky='ew')
    Main()


def stop_live():
    global task, camera
    if camera != None:
        camera.stop()
    task = 0
    draw_button = ttk.Button(right_frame, text="Start Live", command=start_live)
    draw_button.grid(row=47, column=0, columnspan=2, pady=20, padx=10, sticky='ew')
    #Main()
    
    
def start_acquisition():
    global task, bg_average, data
    stop_live()
    data=[[[],[]],[[],[]],[[],[]]]
    cv2.destroyAllWindows()
    task = 2
    root.after(100, Main)


#  Updates the table according to the clicked button   #
def button_intensity():
    global data
    global j
    global tableau_donnees
    j=0
    for i in range (4):
        if data[0][0]!=[]:
            tableau_donnees[i+1][1]= round(data[0][0][i],4)
        else:
            tableau_donnees[i+1][1]='None'
    for k in range (4):
        if data[0][1]!=[]:
            tableau_donnees[k+1][2]=round(data[0][1][k],4)
        else:
            tableau_donnees[k+1][2]='None'
    write_tab(tableau_donnees)    
    
    
def button_phase():
    global data
    global j
    global tableau_donnees
    j=1
    for i in range (4):
        if data[1][0]!=[]:
            tableau_donnees[i+1][1]=round(data[1][0][i],4)
        else:
            tableau_donnees[i+1][1]='None'
    for k in range (4):
        if data[1][1]!=[]:
            tableau_donnees[k+1][2]=round(data[1][1][k],4)
        else:
            tableau_donnees[k+1][2]='None'
    write_tab(tableau_donnees)    
    
    
def button_modulation():
    global data
    global j
    global tableau_donnees
    j=2
    for i in range (4):
        if data[2][0]!=[]:
            tableau_donnees[i+1][1]=round(data[2][0][i],4)
        else:
            tableau_donnees[i+1][1]='None'
    for k in range (4):
        if data[2][1]!=[]:
            tableau_donnees[k+1][2]=round(data[2][1][k],4)
        else:
            tableau_donnees[k+1][2]='None'
    write_tab(tableau_donnees)    

    
def receive_coefficients(coeff_phi, coeff_m):
    global coef_phi, coef_m
    coef_phi, coef_m=coeff_phi, coeff_m
    
    
def calibrate_lifetime():
    global task
    messagebox.showinfo("Lifetime calibration", "Please place the component with the known lifetime")
    task=4
    Main()
    
    
#  In main: capture 150 images and do an average pixel by pixel and set the new 'bg_average' reference
def bg_calibration():
    global task
    messagebox.showinfo("Background calibration", "Please remove the component and place the camera in the same condition as during data capture (it will last a few seconds)")
    draw_button = ttk.Button(right_frame, text="Start Live", command=start_live)
    draw_button.grid(row=47, column=0, columnspan=2, pady=20, padx=10, sticky='ew')
    cv2.destroyAllWindows()
    task = 3
    Main()
    
"""                                 START OF THE GRAPHICS API                                   """  
#  Retrieves images for the API  #
def load_images():
    global logo1_tk, logo2_tk
    logo1 = Image.open("logo3.png")
    logo2 = Image.open("logo2.png")
    w, h = logo1.size
    logo1=logo1.resize((int(w/1.5), int(h/1.5)), Image.LANCZOS)
    w, h = logo2.size
    logo2=logo2.resize((int(w/3), int(h/3)), Image.LANCZOS)
    logo1_tk = ImageTk.PhotoImage(logo1)
    logo2_tk = ImageTk.PhotoImage(logo2)

def tau_from_phi(phi):
    global omega
    return abs(math.tan(math.radians(phi))/(omega))

def tau_from_mod(mod):
    global omega
    if (mod < 1 and mod!=0):
        return(math.sqrt(1 / mod**2 -1)/(omega))
    else:
        return (0)

def lifetime_cancel():
    global coef_m,coef_phi
    coef_m, coef_phi=1, 1
    messagebox.showinfo("Cancel lifetime calibration", "Modulation and phase coefficients have been set to 1")
    
def bg_cancel():
    global bg_average
    bg_average=np.zeros((1008,1008))
    messagebox.showinfo("Cancel background calibration", "Parameter of background calibration has been set to 0")

def display_doc():
    display_pdf("data_sheet_pco.pdf")
        
def display_pdf(pdf_path):
    def show_page():
        y_position = 0
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_tk = ImageTk.PhotoImage(img)
            images.append(img_tk)
            canvas1.create_image(0, y_position, anchor=tk.NW, image=img_tk)
            y_position += pix.height

        canvas1.config(scrollregion=canvas1.bbox(tk.ALL))

    def on_mouse_wheel(event):
        canvas1.yview_scroll(int(-1 * (event.delta / 120)), "units")

    top = tk.Toplevel()
    top.title("Documentation")
    
    pdf_document = fitz.open(pdf_path)
    first_page = pdf_document.load_page(0)
    pix = first_page.get_pixmap()
    page_width = pix.width
    page_height = pix.height
    top.geometry(f"{page_width}x{page_height}")
    
    canvas1 = tk.Canvas(top)
    canvas1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll_y = tk.Scrollbar(top, orient=tk.VERTICAL, command=canvas1.yview)
    scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
    canvas1.configure(yscrollcommand=scroll_y.set)
    canvas1.bind_all("<MouseWheel>", on_mouse_wheel)

    images = []
    show_page()

    top.mainloop()   

def display_coef():
    global coef_phi, coef_m
    messagebox.showinfo("Lifetime calibration parameters", f"phase coeficient = {round(coef_phi,4)} and modulation coefficient = {round(coef_m,4)}")    

def info_bg_calib():
    global bg_average
    if bg_average.all() == np.zeros((1008,1008)).all():
        messagebox.showinfo("Background calibration info", "Background matrix is set to 0 so please start a background calibration if you need a background correction")    
    else:
        messagebox.showinfo("Background calibration info", "Background matrix is ready to be used")    

def create_fitt():
    global task
    task=5
    Main()
    
#  Display the phasor plot with a slide  #
suppr=None 
def plot_phasor():
    global suppr, task, current_image, phasor
    if current_image != None:
        ni=current_image[0]
    val=0
    
    def callback():
        global suppr
        val =slider_plot.get_minmax()
        if suppr and suppr.get_tk_widget().winfo_exists():
            suppr.get_tk_widget().pack_forget()
        mask = (ni >= val)
        real_parts = phasor.real[mask]
        imag_parts = phasor.imag[mask]*-1
        fig=Figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_title("Plot phasor")
        ax.scatter(real_parts, imag_parts, alpha=0.5)
        ax.axis('on')
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Real part")
        ax.set_ylabel("Imaginary part")
        theta = np.linspace(0,np.pi, 100)
        x_circle = 0.5+0.5*np.cos(theta)
        y_circle = 0.5*np.sin(theta)
        ax.plot(x_circle, y_circle, 'r--')
        canvas_fig = FigureCanvasTkAgg(fig, master=top)
        suppr=canvas_fig
        canvas_fig.draw()
        canvas_widget = canvas_fig.get_tk_widget()
        canvas_widget.pack(pady=10, side=tk.LEFT, fill=tk.BOTH, expand=True)
    if task==2 and current_image != None:
        top = tk.Toplevel()
        top.title("Plot phasor")
        real_part = phasor.real
        imaginary_part = phasor.imag*-1
        min_val, max_val, _, _ = cv2.minMaxLoc(ni)
        slider_plot = RangeSlider(top, min_val, max_val, nb=1, width=400, height=100)
        slider_plot.pack(pady=10, padx=20, side=tk.TOP, expand=True)
        slider_plot.set_title('       Diplay pixels with intensity >')
        plot_button = ttk.Button(top, text="Ok", style="small2.TButton",command=callback)
        plot_button.pack(pady=0, side=tk.TOP, expand=True)
        callback()
        top.mainloop()
    else:
        messagebox.showinfo("No acquisition", f"Please start an acquisition before trying to display the phasor plot")
                
    

def add_rectangle_selector2(ax, im, title, fig):
    global rect_selectors
    def onselect(eclick, erelease):
        global x1, x2, y1, y2, top, tableau_donnees
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        for rect_selector in rect_selectors:
            rect_selector.set_active(True)
            rect_selector.extents = (x1, x2, y1, y2) 
            rect_selector.update()
        if x1 == x2 and y1 == y2:
            selected_mean=im[x1][y1]
            selected_min,selected_max, std= selected_mean, selected_mean, selected_mean
        else:
            selected_area = im[y1:y2, x1:x2]
            selected_min, selected_max, _, _ = cv2.minMaxLoc(selected_area)
            selected_mean = np.mean(selected_area)
            std=np.std(selected_area)
        if title != 'intensity':
            tableau_donnees[1][2]=round(selected_mean,4)
            tableau_donnees[2][2]=round(selected_min,4)
            tableau_donnees[3][2]=round(selected_max,4)
            tableau_donnees[4][2]=round(std,4)
            write_tab(tableau_donnees)
            
    rect_selector = RectangleSelector(ax, onselect, useblit=True,
                                      button=[1],  # left click
                                      spancoords='pixels', interactive=True)
    rect_selectors.append(rect_selector)
    fig.canvas.mpl_connect('button_press_event', rect_selector)
    fig.canvas.mpl_connect('button_release_event', rect_selector)

def use_fitting ():
    top = tk.Toplevel()
    top.title("Use fitting equations")
    clean()
    def button_callback(option):
        global task, current_image, rect_selectors
        rect_selectors=[]
        task=6
        if option.get()!='':
            messagebox.showinfo("Acquisition", "Turn on the modulation light and place your component")
            Main()
            if current_image :
                
                slider_intensity = RangeSlider(right_frame, np.min(current_image[0]), np.max(current_image[0]), nb=2, width=400, height=100)
                slider_intensity.grid(row=51, column=0, columnspan=2, rowspan=2, pady=10, padx=20, sticky='nsew')
                slider_intensity.set_title('intensity scale')
                
                k=fitted_name.index(option.get())
                params=fitted_parameters[k]
                fct=fitted_function[k][0]
                print(params)
                for i in range (len(current_image[1])):
                    for j in range (len(current_image[1])):
                        if fitted_function[k][1] == 'phi (deg)':
                            if i==0 and j==0:
                                print (current_image[1][i][j])
                                print(fct(current_image[1][i][j],**params))
                            current_image[1][i][j]=abs(fct(current_image[1][i][j],**params))
                        elif fitted_function[k][1] == 'tau (µs)':
                            if i==0 and j==0:
                                print (current_image[1][i][j])
                                print(fct(current_image[1][i][j],**params))
                            current_image[1][i][j]=abs(fct(10**6*tau_from_phi(current_image[1][i][j]),**params))
                        else:
                            current_image[1][i][j]=abs(fct(fitted_function[k][2](current_image[1][i][j]),**params))
                
                current_image[1]=np.where(np.isinf(current_image[1]), np.nan, current_image[1])
                
                slider_opt = RangeSlider(right_frame, np.nanmin(current_image[1]), np.nanmax(current_image[1]), nb=2, width=400, height=100)
                slider_opt.grid(row=53, column=0, columnspan=2, rowspan=2, pady=10, padx=20, sticky='nsew')
                slider_opt.set_title(f'{option.get()} scale')
                
                def update_display():
                    global current_image
                    clean()
                    vmin,vmax=slider_intensity.get_minmax()
                    plt_to_tk(current_image[0], np.min(current_image[0]), np.max(current_image[0]), np.mean(current_image[0]), 'Intensity', 'gray', vmin=vmin, vmax=vmax)
                    vmin,vmax=slider_opt.get_minmax()
                    plt_to_tk(current_image[1], np.min(current_image[1]), np.max(current_image[1]), np.mean(current_image[1]),str(option.get()), 'viridis', vmin=vmin, vmax=vmax)
                    mean_value=np.mean(current_image[1])
                    min_value=np.nanmin(current_image[1])
                    max_value=np.nanmax(current_image[1])
                    std=np.std(current_image[1])
                    tableau_donnees[1][1]=round(mean_value,4)
                    tableau_donnees[2][1]=round(min_value,4)
                    tableau_donnees[3][1]=round(max_value,4)
                    tableau_donnees[4][1]=round(std,4)
                    data[0][0]= [mean_value,min_value,max_value,std]
                    data[1][0]= [mean_value,min_value,max_value,std]
                    data[2][0]= [mean_value,min_value,max_value,std]
                    write_tab(tableau_donnees)
                update_display()
                slid_button = ttk.Button(right_frame, text="Apply changes", style="small2.TButton",command=update_display)
                slid_button.grid(row=56, column=0, columnspan=2, pady=20, padx=10, sticky='ew')
                top.destroy()
            else:
                # print("A probleme occured, please check your fitting function")
                messagebox.showinfo("Error", "A probleme occured, please check your fitting function")
    ttk.Label(top, text="Select Model:", style='Top.TLabel').grid(row=1, column=1, columnspan=2, padx=10, pady=5, sticky='ew')
    option = ttk.Combobox(top, textvariable="Select a fitting model",style='Top.TCombobox', values=fitted_name, state='readonly')
    option.grid(row=1, column=3, columnspan=2, padx=10, pady=10, sticky='ew')
    ttk.Button(top, text="Calculate", command=lambda:button_callback(option), style='Top.TButton').grid(row=1, column=5, padx=10, pady=10, sticky='ew')
    top.mainloop()
    
    
def adjust_treshold():
    global global_treshold
    top = tk.Toplevel()
    top.title("Plot phasor")
    def callback():
        global global_treshold
        global_treshold=slider_tresh.get_minmax()
        #tresh_label.config(text=f"Current treshold : {global_treshold}")
        top.destroy()
    min_val, max_val = 0,1
    slider_tresh = RangeSlider(top, min_val, max_val, nb=1, width=400, height=100)
    slider_tresh.pack(pady=10, padx=20, side=tk.TOP, expand=True)
    slider_tresh.set_title('       New background treshold >')
    tresh_button = ttk.Button(top, text="Ok", style="small2.TButton",command=callback)
    tresh_button.pack(pady=0, side=tk.TOP, expand=True)
    tresh_label=tk.Label(top, text=f"Current treshold : {round(global_treshold,2)}")
    tresh_label.pack(pady=0, side=tk.BOTTOM, expand=True)

    top.mainloop()

# Main window of the interface  #
root = tk.Tk()
root['bg']='white'
root.title("PCO Camera Flim")
load_images()
root.iconphoto(False, logo1_tk)

menubar = Menu(root)
root.config(menu=menubar)
menufitting = Menu(menubar,tearoff=0)
menubar.add_cascade(label="Fitting", menu=menufitting)
menufitting.add_command(label="Create a fitting", command=create_fitt)
menufitting.add_command(label="Use fitting", command=use_fitting)


menuedition = Menu(menubar,tearoff=0)
menubar.add_cascade(label= "Calibration", menu=menuedition)
menuedition.add_command(label="Background calibration", command=bg_calibration)
menuedition.add_command(label="Adjust background teshold", command=adjust_treshold)
menuedition.add_command(label="Lifetime Calibration", command=calibrate_lifetime)
menuedition.add_command(label="Cancel lifetime calibration", command=lifetime_cancel)
menuedition.add_command(label="Cancel background calibration", command=bg_cancel)

menuhelp = Menu(menubar,tearoff=0)
menubar.add_cascade(label= "Help", menu=menuhelp)
menuhelp.add_command(label="Documentation", command=display_doc)
menuhelp.add_command(label="Show lifetime calibration parameters", command=display_coef)
menuhelp.add_command(label="Background calibration info", command=info_bg_calib)

# configure new style for widgets  #
def setup_styles():
    style = ttk.Style()
    style.theme_use('clam')  # Basic theme
    style.configure('TCombobox', font=('Arial', 10), fieldbackground='white', background='#F0F0F0',selectbackground='white',selectforeground='black', borderwidth=1)
    style.map('TCombobox',fieldbackground=[('readonly', 'white')])
    style.configure('TLabel', font=('Agency FB', 14), background='#FEF8D6', foreground='black')
    style.configure('TButton', font=('Agency FB', 18, 'bold'),foreground='#269577', background='#FAD491', borderwidth=1)
    style.map('TButton', background=[('active', '#F5AA27')])
    style.configure('cross.TButton',font=('Arial',10), foreground='#FFFFFF', background='#DB0202', borderwidth=1, width=1, height=1)
    style.map('cross.TButton', background=[('active', '#DB0202')])
    style.configure('small.TButton', font=('Agency FB', 12, 'bold'),foreground='white', background='#458D7A', borderwidth=1)
    style.map('small.TButton', background=[('active', '#207D64')])
    style.configure('TEntry', font=('Agency FB', 10), background='white', foreground='black',fieldbackground='white')
    style.configure('TCheckbutton',font=('Agency FB', 15),background='#FEF8D6', foreground='black',fieldbackground='#FDF3B4',selectbackground='#F5AA27',selectforeground='black')
    style.map('TCheckbutton', background=[('active', '#FDF3C5')])
    style.configure('small2.TButton', font=('Agency FB', 12, 'bold'),foreground='#269577', background='#FAD491', borderwidth=1)
    style.map('small2.TButton', background=[('active', '#F5AA27')])
    style.configure('small3.TButton', font=('Agency FB', 8), foreground='black', background='#FEF8D6', borderwidth=1)
    style.map('small3.TButton', background=[('active', '#FDF3B4')])
    style.configure('Top.TCombobox', font=('Arial', 10), fieldbackground='white', background='#F0F0F0',selectbackground='white',selectforeground='black', borderwidth=1)
    style.map('Top.TCombobox',fieldbackground=[('readonly', 'white')])
    style.configure('Top.TLabel', font=('Arial', 10), background='#F0F0F0',  padding=5)
    style.configure('Top.TButton',font=('Arial', 10), background='white',  foreground='black', padding=5)
    style.configure('Top.TEntry', font=('Arial', 10),fieldbackground='white', background='#F0F0F0', padding=2)
    style.configure('Top.TFrame',font=('Arial', 10), background='white', padding=10)
    style.configure('Top.TLabelframe',font=('Arial', 10), background='#F0F0F0', padding=5,relief='raised')
    style.configure('Top.TLabelframe.Label',font=('Arial', 10),background='#F0F0F0')
    style.configure('Top.Treeview', font=('Arial', 10),rowheight=25, background='white', fieldbackground='white', borderwidth=1)
    style.map('Top.TButton', background=[('active', '#F0F0F0')])
    style.configure('Top.TCheckbutton',font=('Arial',10),background='#F0F0F0',fieldbackground='#white',selectbackground='lightgrey',selectforeground='black')
    style.map('Top.TCheckbutton', background=[('active', 'lightgray')]) 
    
    
setup_styles()

# Change the size of the main window  #
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")
root.state('zoomed')

# right side frame with user entries  #
right_frame = tk.Frame(root, bg='White')
right_frame.pack(fill=tk.Y, side=tk.RIGHT, expand=False)

# user entry  #
control_frame = tk.Frame(right_frame, bg='#FEF8D6')
control_frame.grid(row=10, column=0, columnspan=2, pady=5, padx=10, sticky='w')
row=0
ttk.Label(control_frame, text="Exposure (s):").grid(row=row, column=0, sticky='e', pady=2, padx=10)
exposure_entry = ttk.Entry(control_frame, style='TEntry')
exposure_entry.configure(font=('Arial',11))
exposure_entry.grid(row=row, column=1, sticky='w', pady=2, padx=10)
row+=1
# ttk.Label(control_frame, text="Modulation source :").grid(row=row, column=0, sticky='e', pady=2, padx=10)
options = ["intern", "extern"]
source_entry = ttk.Combobox(control_frame, values=options, state='readonly')
# source_entry.grid(row=row, column=1, sticky='ew', pady=2, padx=10)
row+=1
ttk.Label(control_frame, text="Wave form :").grid(row=row, column=0, sticky='e', pady=2, padx=10)
options = ["sinusoidal", "rectangular", "none"]
form_entry = ttk.Combobox(control_frame, values=options, state='readonly')
form_entry.configure(font=('Arial',11))
form_entry.grid(row=row, column=1, sticky='w', pady=2, padx=10)
row+=1
ttk.Label(control_frame, text="number of phases :").grid(row=row, column=0, sticky='e', pady=2, padx=10)
options = ["manual shifting","2 phases","4 phases","8 phases","16 phases"]
nb_phase_entry = ttk.Combobox(control_frame, values=options, state='readonly')
nb_phase_entry.configure(font=('Arial',11))
nb_phase_entry.grid(row=row, column=1, sticky='w', pady=2, padx=10)
row+=1
ttk.Label(control_frame, text="Symmetry correction :").grid(row=row, column=0, sticky='e', pady=2, padx=10)
options = ["singular", "twice"]
symmetry_entry = ttk.Combobox(control_frame, values=options, state='readonly')
symmetry_entry.configure(font=('Arial',11))
symmetry_entry.grid(row=row, column=1, sticky='w', pady=2, padx=10)
row+=1
ttk.Label(control_frame, text="Phase order :").grid(row=row, column=0, sticky='e', pady=2, padx=10)
options = ["ascending", "opposite"]
phase_order_entry = ttk.Combobox(control_frame, values=options, state='readonly')
phase_order_entry.configure(font=('Arial',11))
phase_order_entry.grid(row=row, column=1, sticky='w', pady=2, padx=10)
row+=1
ttk.Label(control_frame, text="Selected Tap :").grid(row=row, column=0, sticky='e', pady=2, padx=10)
options = ["both", "tap A", "tap B"]
tap_entry = ttk.Combobox(control_frame, values=options, state='readonly')
tap_entry.configure(font=('Arial',11))
tap_entry.grid(row=row, column=1, sticky='w', pady=2, padx=10)
row+=1
ttk.Label(control_frame, text="Background correction? :").grid(row=row, column=0, sticky='e', pady=2, padx=10)
options = ['Yes', 'No']
bg_entry = ttk.Combobox(control_frame, values=options, state='readonly')
bg_entry.configure(font=('Arial',11))
bg_entry.grid(row=row, column=1, sticky='w', pady=2, padx=10)
row+=1
ttk.Label(control_frame, text="Frequency (Hz):").grid(row=row, column=0, sticky='e', pady=2, padx=10)
freq = ttk.Entry(control_frame)
freq.configure(font=('Arial',11))
freq.grid(row=row, column=1, sticky='w', pady=2, padx=10)
row+=1
apply_filters_var = tk.IntVar()
apply_filters=ttk.Checkbutton(control_frame,text="Apply filters", variable=apply_filters_var)
apply_filters.grid(row=row, column=0, sticky='e', pady=2, padx=10)
tooltip=Tooltip(apply_filters,"Improves signal-to-noise ratio and eliminates defective pixels (Wiener and median Blur filters)")
#  Buttons
draw_button = ttk.Button(right_frame, text="Launch acquisition", command=start_acquisition)
draw_button.grid(row=46, column=0, columnspan=2, pady=10, padx=10, sticky='ew')
draw_button = ttk.Button(right_frame, text="Start Live", command=start_live)
draw_button.grid(row=47, column=0, columnspan=2, pady=10, padx=10, sticky='ew')

control_frame.grid_columnconfigure(1, weight=1)

param=tk.Label(right_frame,text='PARAMETERS',font=('Agency FB', 30, 'bold'), foreground='#269577', bg='White')
param.grid(row=0, column=0, columnspan=2, pady=50, padx=5, sticky='ew')

logo = tk.Label(right_frame,  image = logo2_tk, bg='White')
logo.grid(row=54, column=0, columnspan=3, pady=40, padx=10, sticky='ew')
for i in range(60):
    right_frame.grid_rowconfigure(i, weight=1)
for j in range(10):
    right_frame.grid_columnconfigure(j, weight=1)

#  Frame bottom for tabs  #
bot=tk.Frame(root, relief='raised', borderwidth=1)
bot.pack(padx=0, pady=0, side=tk.BOTTOM, fill=tk.X, expand=True)
tab = tk.Frame(bot, bg="#87C0AB")
tab.pack(padx=5, pady=5, side=tk.BOTTOM, fill=tk.BOTH, expand=True)
button_mod = ttk.Button(bot, text="Modulation", style="small.TButton", command=button_modulation)
button_mod.pack(padx=30, pady=5, side=tk.RIGHT)
button_phi = ttk.Button(bot, text="Phase",style="small.TButton", command=button_phase)
button_phi.pack(padx=30, pady=5, side=tk.RIGHT)
button_intens = ttk.Button(bot, text="Intensity",style="small.TButton", command=button_intensity)
button_intens.pack(padx=30, pady=5, side=tk.RIGHT)
button_export_tableau = ttk.Button(bot, text="Export Table", style="small3.TButton", command=lambda: export_tableau_to_excel(headers,index))
button_export_tableau.pack(padx=5, pady=5, side=tk.LEFT)
phasor_plot_button = ttk.Button(bot, text="Phasor plot", style="small3.TButton",command=plot_phasor)
phasor_plot_button.pack(padx=30, pady=0, side=tk.LEFT)
tk.Label(bot, text="Lifetime from phase and mod (µs):", bg='#E6E6E6', pady=2, padx=2).pack(side=tk.LEFT, padx=10)

tau_phi=tk.Label(bot, text=" ",bg='#E6E6E6', pady=2, padx=2)
tau_phi.pack(side=tk.LEFT, padx=10)
tau_mod=tk.Label(bot, text=" ",bg='#E6E6E6', pady=2, padx=2)
tau_mod.pack(side=tk.LEFT, padx=10)

tk.Label(bot, text="std:", bg='#E6E6E6', pady=2, padx=2).pack(side=tk.LEFT, padx=10)
std=tk.Label(bot, text=" ",bg='#E6E6E6', pady=2, padx=2)
std.pack(side=tk.LEFT, padx=10)


# display the tab in the frame  #
def write_tab(tableau_donnees):  
    for i in range (5):    
        label = tk.Label(tab, text=tableau_donnees[i][0], borderwidth=1, relief="solid", padx=5, pady=5,font=('Agency FB', 15, 'bold'))
        label.grid(row=i, column=0, sticky="nsew", padx=5, pady=5)
    label = tk.Label(tab, text=tableau_donnees[0][1], borderwidth=1, relief="solid", padx=5, pady=5,font=('Agency FB', 15, 'bold'))
    label.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
    label = tk.Label(tab, text=tableau_donnees[0][2], borderwidth=1, relief="solid", padx=5, pady=5,font=('Agency FB', 15, 'bold'))
    label.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
    for i in range(1,5):
            label = tk.Label(tab, text=tableau_donnees[i][1], borderwidth=1, relief="solid", padx=5, pady=5,font=('Agency FB', 13))
            label.grid(row=i, column=1, sticky="nsew", padx=5, pady=5)
    for i in range(1,5):
            label = tk.Label(tab, text=tableau_donnees[i][2], borderwidth=1, relief="solid", padx=5, pady=5,font=('Agency FB', 13))
            label.grid(row=i, column=2, sticky="nsew", padx=5, pady=5)
    # frame can be extended:
    for i in range(5):
        tab.grid_rowconfigure(i, weight=1)
    for j in range(3):
        tab.grid_columnconfigure(j, weight=1)
            
write_tab(tableau_donnees)
       
#  Canvas = center frame with graphs  #
canvas = tk.Canvas(root, bg='white')
canvas.pack(fill=tk.BOTH, expand=True)


#  Label for logo  #
logo_frame= tk.Frame(root).pack(expand=True)
logo = tk.Label(logo_frame,  image = logo1_tk, bg='White')
logo.place(x='10',y='10')

current_image = None

load_settings()
root.protocol("WM_DELETE_WINDOW", on_closing)

# main loop
root.mainloop()



