from PyQt5 import QtWidgets, QtCore, QtGui

import h5py
import serial
import struct
import numpy as np
import time
from numba import jit


class SerialThread(QtCore.QThread):
    internal_error = QtCore.pyqtSignal(object)
    serialecho = QtCore.pyqtSignal(object)
    easyprint = QtCore.pyqtSignal(object)
    devicestatus = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal(bool)
    error = QtCore.pyqtSignal(str)
    def __init__(self, ser):
        super().__init__()
        self.alive = False
        self.serial_read_thread_terminated = False
        self.ser = ser

        self.status = {'saved_counts':0, 'slots_used':0, 'counts_received':0, 'bytes_dropped':False}
        self.counts_received = 0

        self.enable_retention_interval_filter = False
        self.retention_interval = np.int64(1/5E-9)
        self.close_hdf_file = False
        self.hdf_file = None
        self.bytes_dropped = False
        self.save_now = False
        self.saving_records = False
        self.save_temp_when_done = False
        self.dset_records_name = 'records'
        self.dset_num_entries_name = 'total_entries'
        self.blocksize = 10000
        self.pad_byte = bytes(1)
        self.time_mask = 2**52-1
        self.request_status_encoded_command = encode_settings(request_status=True) #A tiny time saver so I don't have to encode each call

    def update_status(self):
        self.save_now = True
        self.write_command(self.request_status_encoded_command)

    def start_saving(self, file_directory):
        self.hdf_file = h5py.File(str(file_directory), 'a')
        record_types = [ ('time', np.int64), ('ch0', np.uint8), ('ch1', np.uint8), ('ch2', np.uint8), ('ch3', np.uint8)]
        if self.dset_records_name in self.hdf_file:
            self.dset_records = self.hdf_file[self.dset_records_name]
            self.dset_num_entries = self.hdf_file[self.dset_num_entries_name]
        else:
            self.dset_records = self.hdf_file.create_dataset(self.dset_records_name, shape=(self.blocksize,), dtype=record_types, maxshape=(None,), chunks=True)
            self.dset_num_entries = self.hdf_file.create_dataset(self.dset_num_entries_name, shape=(1,), dtype=np.int64)

        # self.temp_data = np.empty((self.blocksize, 5), dtype=np.int64)
        self.temp_data = np.empty(self.blocksize, dtype=record_types)
        self.temp_data_idx = 0
        self.saving_records = True
        self.file_directory = file_directory
        self.save_temp_when_done = True

    def stop_saving(self):
        self.saving_records = False
        self.close_hdf_file = True

    def add_data_to_dataset(self, new_data, num_new_entries, dset_records, dset_num_entries, blocksize=10000):
        num_current_entries = dset_num_entries[0]
        free_space = dset_records.size - num_current_entries
        if num_new_entries > free_space:
            dset_records.resize(dset_records.size + max(blocksize, num_new_entries) , axis=0)
        new_total_entries = num_current_entries + num_new_entries
        dset_records[num_current_entries:new_total_entries] = new_data[:num_new_entries]
        dset_num_entries[0] = new_total_entries
        self.status['saved_counts'] = new_total_entries

    def write_command(self, encoded_command):
        self.ser.write(encoded_command)

    def stop(self):
        self.alive = False

    def run(self):
        if self.saving_records:
            self.start_saving(self.file_directory)

        self.alive = True
        self.serial_thread_terminated = False
        remaining_data = np.array((), dtype=np.uint8)

        last_record = np.zeros(5, dtype=np.int64)
        last_record_save = 0
        while self.alive:
            try:
                new_data = self.ser.read(4000)
            except serial.serialutil.SerialException as ex:
                self.alive = False
                self.serial_thread_terminated = True
                self.error.emit(str(ex))
                break
            new_data_arr = np.array(list(new_data), dtype=np.uint8)
            records, records_idx, other_messages, other_messages_idx, remaining_data, out_of_sync = quick_decode(remaining_data, new_data_arr)
            if out_of_sync:
                self.bytes_dropped = True

            if records_idx:
                self.counts_received += records_idx

                if self.saving_records:
                    # Decide here which records will be saved.
                    # create a new array which as the first column be save/not save. The last record of records will always be added as the first entry of
                    # savecheck_array, whether it has been determined to be saved or not. 
                    # I don't like this way of doing it, because I am creating extra arrays, and shuffeling data around when I don't need to. But It seems to work and I don't care enough.
                    if self.enable_retention_interval_filter:
                        records, records_idx, last_record, last_record_save = savecheck(last_record, last_record_save, records, records_idx, self.retention_interval)


                    self.temp_data['time'][self.temp_data_idx:self.temp_data_idx+records_idx] = records[:records_idx, 0]
                    self.temp_data['ch0'][self.temp_data_idx:self.temp_data_idx+records_idx] = records[:records_idx, 1]
                    self.temp_data['ch1'][self.temp_data_idx:self.temp_data_idx+records_idx] = records[:records_idx, 2]
                    self.temp_data['ch2'][self.temp_data_idx:self.temp_data_idx+records_idx] = records[:records_idx, 3]
                    self.temp_data['ch3'][self.temp_data_idx:self.temp_data_idx+records_idx] = records[:records_idx, 4]
                    self.temp_data_idx += records_idx
                    # if the next data dump can possibly overflow the temp_data, then clear the temp data now. the max records in the buffer is about 530
                    if self.temp_data_idx + 550 > self.temp_data.size or self.save_now:
                        self.add_data_to_dataset(self.temp_data, self.temp_data_idx, self.dset_records, self.dset_num_entries, self.blocksize)
                        self.temp_data_idx = 0
                        self.save_now = False
                        self.hdf_file.flush()
            if other_messages_idx:
                for message_arr in other_messages[:other_messages_idx]:
                    message_identifier = message_arr[0]
                    message_bytes = bytes(message_arr[1:msgin_decodeinfo[message_identifier]['message_length']])
                    message = msgin_decodeinfo[message_identifier]['decode_function'](message_bytes)
                    if message_identifier == msgin_identifier['devicestatus']:
                        self.status['counts_received'] = self.counts_received
                        self.status['bytes_dropped'] = self.bytes_dropped
                        self.bytes_dropped = False
                        self.status.update(message)
                        self.devicestatus.emit(self.status)
                    elif message_identifier == msgin_identifier['error']:
                        self.internal_error.emit(message)
                    elif message_identifier == msgin_identifier['echo']:
                        self.serialecho.emit(message)
                    elif message_identifier == msgin_identifier['print']:
                        self.easyprint.emit(message)
            # Close the hdf file if not saving records so the file itself can be modified externally
            if not self.saving_records:
                if self.close_hdf_file:
                    if self.hdf_file: 
                        self.hdf_file.close()
                    self.close_hdf_file = False
        if self.hdf_file: 
            self.hdf_file.close()
        self.finished.emit(self.serial_thread_terminated)
        
@jit(nopython=True, cache=True)
def savecheck(last_record, last_record_save, records, records_idx, retention_interval):
    save_array = np.zeros((600, 6), dtype=np.int64)
    save_array[0, 1:] = last_record
    if last_record_save:
        save_array[0, 0] = 1
    save_array[1:records_idx + 1, 1:] = records[:records_idx]
    for idx in range(1, records_idx):
        if save_array[idx, 1] - save_array[idx-1, 1] <= retention_interval:
            save_array[idx-1, 0] = 1
            save_array[idx, 0] = 1
    save_idxs = (save_array[:, 0] == 1)
    last_record = records[records_idx]
    last_record_save = save_array[idx, 0]
    saved_records = save_idxs.sum()
    return save_array[save_idxs, 1:], saved_records, last_record, last_record_save

@jit(nopython=True, cache=True)
def quick_decode(remaining_data, new_data):
    #both inputs are arrays
    data = np.concatenate((remaining_data, new_data))#.astype(np.int64)
    records_idx = 0
    records = np.zeros((600, 5), dtype=np.int64)
    other_messages_idx = 0
    other_messages = np.zeros((20, 9), dtype=np.uint8)
    out_of_sync = False
    N = data.size
    idx = 0
    # find out how many bytes are in the message
    if N != 0:
        while True:
            key = data[idx]
            idx += 1
            if key == 204:
                message_bytes = 14
            elif key == 203:
                message_bytes = 4
            elif key == 201:
                message_bytes = 8
            elif key == 200:
                message_bytes = 2
            elif key == 202:
                message_bytes = 8
            else:
                # If out of sync, just discard bytes until a valid key is found.
                out_of_sync = True
                if idx == N:
                    break
                else:
                    continue
            #Check if the whole message is in the remaining array
            if idx + message_bytes > N:
                idx -= 1 #set the index back one so the key is included in the remaining data
                break
            #Read the whole message
            message = data[idx:idx+message_bytes]
            idx += message_bytes

            if key == 204:
                records[records_idx, 1] = (message[6] >> 4) & 0b1
                records[records_idx, 2] = (message[6] >> 5) & 0b1
                records[records_idx, 3] = (message[6] >> 6) & 0b1
                records[records_idx, 4] = (message[6] >> 7) & 0b1
                records[records_idx, 0] = (int(message[6] & 0b00001111) << 48) | (int(message[5]) << 40) | (int(message[4]) << 32) | (int(message[3]) << 24) | (int(message[2]) << 16) | (int(message[1]) << 8) | int(message[0])
                records_idx += 1
                records[records_idx, 1] = (message[13] >> 4) & 0b1
                records[records_idx, 2] = (message[13] >> 5) & 0b1
                records[records_idx, 3] = (message[13] >> 6) & 0b1
                records[records_idx, 4] = (message[13] >> 7) & 0b1
                records[records_idx, 0] = (int(message[13] & 0b00001111) << 48) | (int(message[12]) << 40) | (int(message[11]) << 32) | (int(message[10]) << 24) | (int(message[9]) << 16) | (int(message[8]) << 8) | int(message[7])
                records_idx += 1
            else:
                other_messages[other_messages_idx, 0] = key
                other_messages[other_messages_idx, 1:message_bytes+1] = message[:message_bytes]
                if other_messages_idx < 19:
                    other_messages_idx += 1
            # If the data array was the perfect length, return
            if idx == N:
                break
    return records, records_idx, other_messages, other_messages_idx, data[idx:], out_of_sync

def decode_internal_error(message):
    ''' Messagein identifier:  1 byte: 200
    Message format:                     BITS USED   FPGA INDEX.
    tags:               1 byte  [0]     2 bits      [0+:2]      unsigned int.
        invalid_identifier_received     1 bit       [0]
        timeout_waiting_for_full_msg    1 bit       [1]  
        received_message_not_forwarded  1 bit       [2]  
    error information:  1 byte  [1]     8 bits      [8+:8]     unsigned int.

    The 'error_info' represents the "device_index" for the received message, which basically says where the meassage should have headed in the FPGA.
    '''
    tags, =         struct.unpack('<Q', message[0:1] + bytes(7))
    error_info, =   struct.unpack('<Q', message[1:2] + bytes(7))
    invalid_identifier_received_tag =       (tags >> 0) & 0b1        
    timeout_waiting_for_msg_tag =           (tags >> 1) & 0b1     
    received_message_not_forwarded_tag =    (tags >> 2) & 0b1 
    invalid_identifier_received =       decode_lookup['invalid_identifier'][invalid_identifier_received_tag]
    timeout_waiting_for_msg =           decode_lookup['msg_receive_timeout'][timeout_waiting_for_msg_tag]
    received_message_not_forwarded =    decode_lookup['msg_not_forwarded'][received_message_not_forwarded_tag]
    return {'invalid_identifier_received':invalid_identifier_received, 'timeout_waiting_to_receive_message':timeout_waiting_for_msg, 'received_message_not_forwarded':received_message_not_forwarded, 'error_info':error_info}

def decode_serialecho(message):
    ''' Messagein identifier:  1 byte: 201
    Message format:                     BITS USED   FPGA INDEX.
    echoed byte:        1 bytes [0:1]   8 bits      [0+:8]     
    device version:     7 bytes [1:8]   56 bits     [8+:56]    '''
    echoed_byte = message[0:1]
    try:
        device_version = message[1:8].decode()
        unprintable_byte = False
    except UnicodeDecodeError as err:
        device_version = message[1:8].decode(errors='ignore')
        unprintable_byte = True
    return {'echoed_byte':echoed_byte, 'device_version':device_version, 'unprintable_byte':unprintable_byte}

def decode_easyprint(message):
    ''' Messagein identifier:  1 byte: 202
    Message format:                     BITS USED   FPGA INDEX.
    printed message:    8 bytes [0:3]   64 bits     [0+:64]     '''
    binary_representation = []
    for letter in message[::-1]:
        binary_representation.append('{:08b} '.format(letter))
    return {'printed':''.join(binary_representation)}

def decode_pulserecord(message):
    ''' Messagein identifier:  1 byte: 204
    Message format:                     BITS USED   FPGA INDEX.
    record A:           7 bytes [0:7]   56 bits     [0+:56]     unsigned int.
        record time                     52 bits     [0+:52]     unsigned int.
        channel 0 tag                   1 bit       [52]
        channel 1 tag                   1 bit       [53]
        channel 2 tag                   1 bit       [54]
        channel 3 tag                   1 bit       [55]
    record B:           7 bytes [7:14]  56 bits     [56+:56]     unsigned int.
        record time                     52 bits     [56+:52]     unsigned int.
        channel 0 tag                   1 bit       [108]
        channel 1 tag                   1 bit       [109]
        channel 2 tag                   1 bit       [110]
        channel 3 tag                   1 bit       [111]
    '''
    record_A, = struct.unpack('<Q', message[0:7] + bytes(1))
    record_A_time = record_A & 2**52-1 # mask of 52 1's
    record_A_ch0_tag = (record_A >> 52) & 0b1  
    record_A_ch1_tag = (record_A >> 53) & 0b1  
    record_A_ch2_tag = (record_A >> 54) & 0b1  
    record_A_ch3_tag = (record_A >> 55) & 0b1
    record_B, = struct.unpack('<Q', message[7:14] + bytes(1))      
    record_B_time = record_B & 2**52-1 # mask of 52 1's
    record_B_ch0_tag = (record_B >> 52) & 0b1  
    record_B_ch1_tag = (record_B >> 53) & 0b1  
    record_B_ch2_tag = (record_B >> 54) & 0b1  
    record_B_ch3_tag = (record_B >> 55) & 0b1  
    return (record_A_time, record_A_ch0_tag, record_A_ch1_tag, record_A_ch2_tag, record_A_ch3_tag), (record_B_time, record_B_ch0_tag, record_B_ch1_tag, record_B_ch2_tag, record_B_ch3_tag)
    # return {'record_A:':[record_A_time, record_A_ch0_tag, record_A_ch1_tag, record_A_ch2_tag, record_A_ch3_tag], 'record_B:':[record_B_time, record_B_ch0_tag, record_B_ch1_tag, record_B_ch2_tag, record_B_ch3_tag]}

def decode_devicestatus(message):
    ''' Messagein identifier:  1 byte: 203
    Message format:                     BITS USED   FPGA INDEX.
    FIFO_slots_used:      bytes [0:3]   25 bits     [0+:25]     unsigned int.
    '''
    slots_used, = struct.unpack('<Q', message[0:4] + bytes(4))
    return {'slots_used':slots_used}

#### encode
def encode_echo(byte_to_echo):
    ''' Messageout identifier:  1 byte: 150
    Message format:                             BITS USED   FPGA INDEX.
    byte_to_echo:               1 byte  [0:18]  8 bits     [0+:8]  
    '''    
    message_identifier = struct.pack('B', msgout_identifier['echo'])
    return message_identifier + byte_to_echo

def encode_general_debug(message):
    ''' Messageout identifier:  1 byte: 151
    Message format:                             BITS USED   FPGA INDEX.
    general_putpose_input:      8 bytes [0:8]   64 bits     [0+:64]     unsigned int.
    '''
    message_identifier =    struct.pack('B', msgout_identifier['general_input'])
    message =               struct.pack('<Q', message)[:8]
    return message_identifier + message

def encode_settings(enable_record=None, enable_send_record=None, holdoff_time=None, request_status=False, purge_memory=False, zero_pulse_timer=False, reset_device=False):
    ''' Messageout identifier:  1 byte: 152
    Message format:                             BITS USED   FPGA INDEX.
    Tag_settings:               1 byte  [0]     8 bits      [0+:8]       unsigned int.
        enable_record                           1bit        [0]
        update_enable_record                    1bit        [1]
        enable_send_record                      1bit        [2]
        update_enable_send_record               1bit        [3]
        request_status                          1bit        [4]
        purge_memory                            1bit        [5]
        zero_pulse_timer                        1bit         [6]
        reset_device                            1bit        [7]
    holdoff_time                4 bytes         28bits      [7+:28]
        update_holdoff_time                     1bit        [35]

    Note, this is now a mix of setting and action requests.
    '''
    holdoff_time_val = 0
    if holdoff_time         is not None: holdoff_time_val = holdoff_time | (1 << 28)
    enable_record_tag       = encode_lookup['enable_record'][enable_record] << 0
    enable_send_record_tag  = encode_lookup['enable_send_record'][enable_send_record] << 2
    request_status_tag      = encode_lookup['request_status'][request_status] << 4
    purge_memory_tag        = encode_lookup['purge_memory'][purge_memory] << 5
    zero_pulse_timer_tag    = encode_lookup['zero_pulse_timer'][zero_pulse_timer] << 6
    reset_device_tag        = encode_lookup['reset_device'][reset_device] << 7
    tags = enable_record_tag | enable_send_record_tag | request_status_tag | purge_memory_tag | zero_pulse_timer_tag | reset_device_tag
    message_identifier =    struct.pack('B', msgout_identifier['settings'])
    tags =                  struct.pack('<Q', tags)[:1]
    holdoff_time =          struct.pack('<Q', holdoff_time_val)[:4]
    return message_identifier + tags + holdoff_time


msgin_decodeinfo = {
    200:{'message_length':3, 'decode_function':decode_internal_error},
    201:{'message_length':9, 'decode_function':decode_serialecho},
    202:{'message_length':9, 'decode_function':decode_easyprint},
    203:{'message_length':5, 'decode_function':decode_devicestatus},
    204:{'message_length':15, 'decode_function':decode_pulserecord}}

msgin_identifier = {
    'error':200,
    'echo':201,
    'print':202,
    'devicestatus':203,
    'pulserecord':204}

decode_lookup = {
    'invalid_identifier':{1:True, 0:False},
    'msg_not_forwarded':{1:True, 0:False},
    'msg_receive_timeout':{1:True, 0:False}
}

msgout_identifier = {
    'echo':150,
    'general_input':151,
    'settings':152
}

encode_lookup = {
    'request_status':{True:1, False:0},
    'reset_device':{True:1, False:0},
    'purge_memory':{True:1, False:0},
    'zero_pulse_timer':{True:1, False:0},
    'enable_record':{True:0b11, False:0b10, None:0b00},
    'enable_send_record':{True:0b11, False:0b10, None:0b00}
}


def print_bytes(bytemessage):
    print('Message:')
    # for letter in instruction[:1:-1]:
    for letter in bytemessage[::-1]:
        print('{:08b}'.format(letter), end =" ")
    print('')
