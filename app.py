from tkinter import *
from os import listdir
from os.path import isfile, join
from helper_functions import *
from scan_controller import ScanController

only_files = [f for f in listdir('images') if isfile(join('images', f))]
for idx, val in enumerate(only_files):
    only_files[idx] = 'images/' + val
scan_results = [0] * len(only_files)

print(only_files)
print(scan_results)

# create the canvas, size in pixels
canvas = Canvas(width=1250, height=800, bg='black')

# pack the canvas into a frame/forms
canvas.pack(expand=YES, fill=BOTH)

scan_controller = ScanController(canvas, only_files, scan_results)

# setup key-binds
canvas.bind("<a>", lambda event: scan_controller.accept_scan())
canvas.bind("<d>", lambda event: scan_controller.deny_scan())
canvas.bind("<Left>", lambda event: scan_controller.prev_scan())
canvas.bind("<Right>", lambda event: scan_controller.next_scan())

# run it ...
canvas.focus_set()
mainloop()
