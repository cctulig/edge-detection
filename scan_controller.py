from tkinter import PhotoImage, Canvas, NW

from PIL import Image, ImageDraw, ImageTk

from helper_functions import open_next_image, fill_result_bar


class ScanController:
    def __init__(self, canvas: Canvas, only_files: list, scan_results: list):
        self.info_image = None
        self.canvas = canvas
        self.only_files = only_files
        self.scan_results = scan_results
        self.current_index = 0
        self.refresh_info()
        self.current_image = open_next_image(self.canvas, self.only_files[self.current_index])

    def next_scan(self):
        if self.current_index < len(self.only_files) - 1:
            self.current_index += 1
        self.rescan()

    def prev_scan(self):
        if self.current_index > 0:
            self.current_index -= 1
        self.rescan()

    def accept_scan(self):
        self.scan_results[self.current_index] = 1
        self.refresh_info()

    def deny_scan(self):
        self.scan_results[self.current_index] = 2
        self.refresh_info()

    def rescan(self):
        self.refresh_info()
        self.current_image = open_next_image(self.canvas, self.only_files[self.current_index])

    def refresh_info(self):
        self.canvas.delete('info')
        blank_image = Image.new('RGBA', (1000, 190), 'white')
        img_draw = ImageDraw.Draw(blank_image)
        img_draw.rectangle((0, 0, 600, 150), outline=fill_result_bar(self.scan_results[self.current_index]), width=10)
        img_draw.rectangle((5, 5, 100, 30), fill='green')
        img_draw.text((10, 10), 'Accept [A]', fill='white')
        img_draw.rectangle((150, 5, 250, 30), fill='red')
        img_draw.text((160, 10), 'Deny [D]', fill='white')
        img_draw.text((10, 40), 'Current Scan: {0}, of Total Scans: {1}\n'
                                'Use <- and -> to navigate between scans.'.format(self.current_index + 1, len(self.only_files)), fill='black')
        img_draw.text((10, 100), 'Successful scans: {0} / {2}\nFailed scans: {1} / {2}'.format(self.scan_results.count(1), self.scan_results.count(2), len(self.only_files)), fill='black')
        self.info_image = ImageTk.PhotoImage(blank_image)
        self.canvas.create_image(10, 610, image=self.info_image, anchor=NW, tags='info')


