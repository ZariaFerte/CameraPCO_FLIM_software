# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:32:11 2024

@author: Zaria Ferté
"""
import tkinter as tk

class Tooltip:
    def __init__(self, widget, text, line_length=30):
        self.widget = widget
        self.text = self.wrap_text(text, line_length)
        self.tooltip_window = None
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def wrap_text(self, text, line_length):
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            if sum(len(w) for w in current_line) + len(word) + len(current_line) - 1 < line_length:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]

        lines.append(' '.join(current_line))
        return '\n'.join(lines)

    def show_tooltip(self, event):
        x = event.x_root + 10
        y = event.y_root + 10
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="#FAD491", font=('Arial',8), relief="solid", borderwidth=1, justify='left')
        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None