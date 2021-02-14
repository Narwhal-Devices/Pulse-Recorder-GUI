from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox
import numpy as np
import time
import pathlib
import serial
import serial.tools.list_ports
import sys
import qdarkstyle

import pulse_recorder_mainwindow_design
import pulse_recorder_additional_classes as prExtras

"""
To run code:
After a standard anaconda python install, some packages will need to be indtalled manually.
To install packages, open a command prompt (Run as administrator if anaconda is installed with admin privileges).
In command prompt, type:

conda activate
conda install pyserial
conda install qdarkstyle

If it complains about other packages that are missing, install theres in a similar way. Check for them using 'conda search xxx' or 'pip search xxx'

Then you should be able to run it. >> python pulse_recorder.py
"""



"""
For code development:
To generate gui, type in terminal:
pyuic5 pulse_recorder_mainwindow_design.ui -o pulse_recorder_mainwindow_design.py

# Make the installer:
# pyinstaller pulse_recorder.py -n "Pulse Recorder" --windowed --noconfirm --clean --icon="icon_master.ico" --add-data="icon_master.ico;."
# pyinstaller pulse_recorder.py -n "Pulse Recorder" --onefile --windowed --noconfirm --clean --icon="icon_master.ico" --add-data="icon_master.ico;."
"""


class MainWindow(QtWidgets.QMainWindow, pulse_recorder_mainwindow_design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Narwhal Devices - Pulse Recorder v1.0.0')
        self.setWindowIcon(QtGui.QIcon('icon_master.ico'))
        self.menubar.setNativeMenuBar(False) #workaround for a bug to make menubar show
        self.statusLabel = QtWidgets.QLabel()
        self.statusbar.addPermanentWidget(self.statusLabel)
        self.update_statuslabel(connection='Not connected', saving='Not saving records')

        # Hide some buttons I used during development
        self.btnEnableSend.hide()
        self.btnDisableSend.hide()

        #connecting the active elements
        self.btnFileSelect.clicked.connect(self.set_file_select)
        self.btnStartSaving.clicked.connect(self.start_saving)
        self.btnStopSaving.clicked.connect(self.stop_saving)
        self.btnZeroTimer.clicked.connect(self.zero_timer)
        self.btnPurgeMemory.clicked.connect(self.purge_memory)
        self.lineEditHoldoff.editingFinished.connect(self.set_holdoff)
        self.btnEnableSend.clicked.connect(self.enable_send)
        self.btnDisableSend.clicked.connect(self.disable_send)
        self.checkBoxRetention.stateChanged.connect(self.retention_enable)
        self.lineEditRetention.editingFinished.connect(self.set_retention)

        #setup serial port
        self.ser = serial.Serial()
        self.ser.timeout = 0.1          #block for 100ms second
        self.ser.writeTimeout = 1     #timeout for write
        self.ser.baudrate = 12000000
        self.ser.port = 'COM6'

        # setting some default values
        self.file_directory = pathlib.Path.home()/'Desktop/pulse_record.hdf'
        self.btnStopSaving.setEnabled(False)
        self.last_counts = (0, 0)
        self.last_holdoff = 10E-9

        #Setting up serial read thread
        self.serial_thread = prExtras.SerialThread(self.ser)

        self.serial_thread.finished.connect(self.callback_finished)
        self.serial_thread.error.connect(self.callback_error)

        self.serial_thread.serialecho.connect(self.callback_echo)
        self.serial_thread.easyprint.connect(self.callback_easyprint)
        self.serial_thread.devicestatus.connect(self.callback_devicestatus)
        self.serial_thread.internal_error.connect(self.callback_internalerror)

        self.authantication_byte = None
        self.valid_ports = []

        self.status_timer = QtCore.QTimer()
        self.status_timer.setInterval(500)
        self.status_timer.timeout.connect(self.serial_thread.update_status)
        
        self.connect_serial()

    def connect_serial(self):
        if self.valid_ports:
            # now try a port
            comport = self.valid_ports.pop(0)
            self.ser.port = comport.device
            try:
                self.ser.open()
            except Exception as ex:
                #if port throws an error on open, wait a bit, then try a new one
                QtCore.QTimer.singleShot(100, self.connect_serial)
                return
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.serial_thread.start()
            self.tested_authantication_byte = np.random.bytes(1)
            self.serial_thread.write_command(prExtras.encode_echo(self.tested_authantication_byte))
            QtCore.QTimer.singleShot(1000, self.check_authantication_byte)
        else:
            # if there are no ports left in the list, add any valid ports to the list  
            comports = list(serial.tools.list_ports.comports())
            for comport in comports:
                if 'vid' in vars(comport) and 'pid' in vars(comport):
                    if vars(comport)['vid'] == 1027 and vars(comport)['pid'] == 24592:
                        self.valid_ports.append(comport)
            if self.valid_ports:
                self.connect_serial()
            else:
                QtCore.QTimer.singleShot(1000, self.connect_serial)

    def check_authantication_byte(self):
        if self.authantication_byte == self.tested_authantication_byte:
            if self.serial_thread.saving_records:
                self.update_statuslabel(saving='Saving records')    
            self.update_statuslabel(connection=f'Connected to {self.ser.port}')
            self.set_holdoff()
            self.serial_thread.write_command(prExtras.encode_settings(enable_record=True, enable_send_record=True))
            self.status_timer.start()
        else:
            self.safe_close_serial_thread()
            QtCore.QTimer.singleShot(1000, self.connect_serial)

    def safe_close_serial_thread(self):
        self.serial_thread.stop()
        self.serial_thread.wait()
        self.ser.close()
        self.update_statuslabel(connection='Not connected', saving='Not saving records')    

    def update_statuslabel(self, saving=None, connection=None):
        if saving: self.status_saving = saving
        if connection: self.status_connection = connection
        self.statusLabel.setText(f'{self.status_saving}    {self.status_connection}')

    def set_file_select(self):
        caption = 'Set Save File'
        file_directory, fileformat = QtWidgets.QFileDialog.getSaveFileName(self, caption=caption, directory=str(self.file_directory), filter='Hierarchical Data Format (*.hdf)', options=QtWidgets.QFileDialog.DontConfirmOverwrite)
        if file_directory:
            self.file_directory = pathlib.Path(file_directory)
            if self.file_directory.is_file():
                buttonReply = QMessageBox.question(self, 'File already exists', 'The selected file already exists, so new data will be appended.\nContinue using this file?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if buttonReply == QtWidgets.QMessageBox.No:
                    self.set_file_select()
            self.lineEditSaveFile.setText(str(self.file_directory))
            return True

    def retention_enable(self, state):
        if state == 2:
            # enable
            self.lineEditRetention.setEnabled(True)
            self.serial_thread.enable_retention_interval_filter = True
        else:
            #disable
            self.lineEditRetention.setEnabled(False)
            self.serial_thread.enable_retention_interval_filter = False

    def start_saving(self):
        if self.lineEditSaveFile.text() == '':
            if not self.set_file_select():
                return
        self.btnStopSaving.setEnabled(True)
        self.btnStartSaving.setEnabled(False)
        self.serial_thread.start_saving(self.file_directory)
        if self.serial_thread.alive:
            self.update_statuslabel(saving='Saving records')

    def stop_saving(self):
        self.serial_thread.stop_saving()
        self.btnStopSaving.setEnabled(False)
        self.btnStartSaving.setEnabled(True)
        self.update_statuslabel(saving='Not saving records')

    def zero_timer(self):
        command = prExtras.encode_settings(zero_pulse_timer=True)
        self.serial_thread.write_command(command)

    def purge_memory(self):
        command = prExtras.encode_settings(purge_memory=True)
        self.serial_thread.write_command(command)

    def enable_send(self):
        command = prExtras.encode_settings(enable_send_record=True)
        self.serial_thread.write_command(command)

    def disable_send(self):
        command = prExtras.encode_settings(enable_send_record=False)
        self.serial_thread.write_command(command)

    def set_holdoff(self):
        txt = self.lineEditHoldoff.text()
        num_str = ''.join(i for i in txt if i.isdigit() or i =='.')
        if 'mi' in txt:
            power = -6
        elif 'm' in txt:
            power = -3
        elif 'μ' in txt:
            power = -6
        elif 'u' in txt:
            power = -6
        elif 'n' in txt:
            power = -9
        elif 'p' in txt:
            power = -12
        else:
            power = 0
        try:
            num = float(num_str)*10**(power)
        except ValueError:
            num = self.last_holdoff
        self.last_holdoff = num
        num = min(num, 1.3)
        num = max(num, 10E-9)
        cycles = round(num/5E-9)
        secs = cycles*5E-9
        if secs >= 1.0:
            disp_txt = '{:.9f}'.format(secs).rstrip('0').rstrip('.') + 's'
        elif secs >= 1E-3:
            disp_txt = '{:.6f}'.format(secs*1E3).rstrip('0').rstrip('.') + 'ms'
        elif secs >= 1E-6:
            disp_txt = '{:.3f}'.format(secs*1E6).rstrip('0').rstrip('.') + 'μs'
        else:
            disp_txt = '{:d}ns'.format(int(secs*1E9))
        self.lineEditHoldoff.setText(disp_txt)
        command = prExtras.encode_settings(holdoff_time=int(cycles-2))
        self.serial_thread.write_command(command)
        self.lineEditHoldoff.clearFocus()

    def set_retention(self):
        txt = self.lineEditRetention.text()
        num_str = ''.join(i for i in txt if i.isdigit() or i =='.')
        if 'mi' in txt:
            power = -6
        elif 'm' in txt:
            power = -3
        elif 'μ' in txt:
            power = -6
        elif 'u' in txt:
            power = -6
        elif 'n' in txt:
            power = -9
        elif 'p' in txt:
            power = -12
        else:
            power = 0
        try:
            num = float(num_str)*10**(power)
        except ValueError:
            num = 1.0
        cycles = round(num/5E-9)
        secs = cycles*5E-9
        if secs >= 1.0:
            disp_txt = '{:.9f}'.format(secs).rstrip('0').rstrip('.') + 's'
        elif secs >= 1E-3:
            disp_txt = '{:.6f}'.format(secs*1E3).rstrip('0').rstrip('.') + 'ms'
        elif secs >= 1E-6:
            disp_txt = '{:.3f}'.format(secs*1E6).rstrip('0').rstrip('.') + 'μs'
        else:
            disp_txt = '{:d}ns'.format(int(secs*1E9))
        self.lineEditRetention.setText(disp_txt)
        self.serial_thread.retention_interval = np.int64(cycles)
        self.lineEditRetention.clearFocus()

    def callback_finished(self, serial_thread_terminated):
        if serial_thread_terminated:
            self.status_timer.stop()
            self.ser.close()
            self.update_statuslabel(connection='Not connected', saving='Not saving records')
            self.connect_serial()
    
    def callback_error(self, error):
        self.statusbar.showMessage(error, 5000)

    def callback_echo(self, message):
        self.authantication_byte = message['echoed_byte']
        self.statusbar.showMessage('Firmware version: {}'.format(message['device_version']), 10000)

    def callback_easyprint(self, message):
        pass
    
    def callback_internalerror(self, message):
        pass

    def callback_devicestatus(self, message):
        current_rate = (message['counts_received'] - self.last_counts[0])/0.5 + (message['slots_used'] - self.last_counts[1])*2/0.5
        self.last_counts = (message['counts_received'], message['slots_used'])
        self.labelCountRateIndicator.setText('{:,} cps'.format(int(current_rate)))
        self.labelSavedCounts.setText('{:,}'.format(message['saved_counts']))
        self.labelMemoryIndicator.setText('{:,}\n/32,000,000'.format(message['slots_used']*2))
        self.barMemoryIndicator.setValue(message['slots_used']/160000)
        if message['bytes_dropped']:
            self.statusbar.showMessage('Bytes dropped', 1000)
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    # app.setStyle('Fusion')
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
