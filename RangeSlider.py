# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:26:38 2024

@author: Zaria Ferté
"""

import tkinter as tk

class RangeSlider(tk.Canvas):
    def __init__(self, parent, min_val, max_val, nb, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.current_min = min_val
        self.current_max = max_val
        self.nb= nb
        self.width = kwargs.get('width', 400)
        self.height = kwargs.get('height', 80)
        self.title = None
        self.config(width=self.width, height=self.height, bg='white')
        self.line = self.create_line(15, self.height // 2, self.width - 15, self.height // 2, fill='gray', width=2)
        self.min_handle = self.create_oval(50, 50, 20, 20, fill='#269577', outline='#269577')
        if self.nb==2 :
            self.max_handle = self.create_oval(0, 0, 20, 20, fill='#269577', outline='#269577')
        self.place_handles()
        self.create_graduations()

        self.bind("<B1-Motion>", self.move_handle)
        self.bind("<ButtonPress-1>", self.on_click)

    def place_handles(self):
        min_pos = self.value_to_position(self.current_min)
        max_pos = self.value_to_position(self.current_max)
        self.coords(self.min_handle, min_pos - 5, self.height // 2 - 10, min_pos + 15, self.height // 2 + 10)
        if self.nb==2 :
            self.coords(self.max_handle, max_pos - 15, self.height // 2 - 10, max_pos + 5, self.height // 2 + 10)
        self.update_labels()

    def create_graduations(self):
        for i in range(11):  # 10 graduations + 1 for max
            x = 15 + i * (self.width - 30) // 10
            self.create_line(x, self.height // 2 - 10, x, self.height // 2 + 10, fill='black')
            value = self.min_val + i * (self.max_val - self.min_val) / 10
            self.create_text(x, self.height // 2 + 20, text=f"{round(value,2)}", fill='black')

    def update_labels(self):
        min_pos = self.value_to_position(self.current_min)
        max_pos = self.value_to_position(self.current_max)

        if hasattr(self, 'min_label'):
            self.delete(self.min_label)
        self.min_label = self.create_text(min_pos, self.height // 2 - 20, text=f"{round(self.current_min,2)}", fill='#269577')

        if hasattr(self, 'max_label'):
            self.delete(self.max_label)
        self.max_label = self.create_text(max_pos, self.height // 2 - 20, text=f"{round(self.current_max,2)}", fill='#269577')
        
        if self.title is not None:
            #title = self.value_to_position(self.title)
            if hasattr(self, 'title_label'):
                self.delete(self.title_label)
            self.title_label = self.create_text(70, self.height // 2 - 40, text=f"{self.title}", fill='black')
            
    def value_to_position(self, value):
        ratio=0
        if (self.max_val - self.min_val) != 0:
            ratio = (value - self.min_val) / (self.max_val - self.min_val)
        return 10 + ratio * (self.width - 20)

    def position_to_value(self, pos):
        ratio = (pos - 10) / (self.width - 20)
        return self.min_val + ratio * (self.max_val - self.min_val)

    def move_handle(self, event):
        handle = self.handle_under_cursor(event)
        if handle:
            new_pos = min(max(event.x, 10), self.width - 10)
            if self.nb==2: 
                if handle == self.min_handle and new_pos < self.coords(self.max_handle)[0] - 10:
                    self.coords(handle, new_pos - 10, self.height // 2 - 10, new_pos + 10, self.height // 2 + 10)
                    self.current_min = self.position_to_value(new_pos)
                if handle == self.max_handle and new_pos > self.coords(self.min_handle)[0] + 10:
                    self.coords(handle, new_pos - 10, self.height // 2 - 10, new_pos + 10, self.height // 2 + 10)
                    self.current_max = self.position_to_value(new_pos)
            elif self.nb==1:
                self.coords(handle, new_pos - 10, self.height // 2 - 10, new_pos + 10, self.height // 2 + 10)
                self.current_min = self.position_to_value(new_pos)
            self.update_labels()
            

    def set_on_value_change(self, callback):
        self.on_value_change = callback

    def handle_under_cursor(self, event):
        if self.coords(self.min_handle)[0] <= event.x <= self.coords(self.min_handle)[2]:
            return self.min_handle
        elif self.nb==2 :
            if self.coords(self.max_handle)[0] <= self.coords(self.max_handle)[2]:
                return self.max_handle
        return None

    def on_click(self, event):
        handle = self.handle_under_cursor(event)
        if handle:
            self.bind("<B1-Motion>", self.move_handle)
        else:
            self.unbind("<B1-Motion>")
    
    def set_title(self, title):
        self.title = title
        self.update_labels()
    
    def get_minmax(self):
        if self.nb==2:
            return(self.current_min,self.current_max)
        return(self.current_min)