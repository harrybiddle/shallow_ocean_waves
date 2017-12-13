# -*- coding: utf-8 -*-
"""
Demonstrates very basic use of ImageItem to display image data inside a ViewBox.
"""

from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime

class Video():

    def __init__(self, data, update_callback):
        self.data = data
        self.update_callback = update_callback

    def _create_app(self):
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsLayoutWidget()
        self.win.show()
        self.image = pg.ImageItem()
        self.view = self.win.addViewBox()
        self.view.addItem(self.image)
        nj, ni = self.data.shape
        self.view.setRange(QtCore.QRectF(0, 0, nj, ni))

    def _update_data(self):
        self.update_callback()
        self.image.setImage(self.data)
        QtCore.QTimer.singleShot(1, self._update_data)

    def _start_qt_event_loop(self):
        app = self.app
        QtGui.QApplication.instance().exec_()

    def show(self):
        self._create_app()
        self._update_data()
        self._start_qt_event_loop()

if __name__ == '__main__':
    # generate data
    all_data = np.random.normal(size=(20, 120, 120), loc=1024, scale=64).astype(np.uint16)
    i = 0
    data = all_data[i]

    # callback that updates the data
    def update_callback():
        global i, data, all_data
        i = (i+1) % all_data.shape[0]
        np.copyto(dst=data, src=all_data[i])

    v = Video(data, update_callback)
    v.show()
