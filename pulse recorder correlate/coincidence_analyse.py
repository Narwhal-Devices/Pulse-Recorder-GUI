from pylab import *
import h5py
import time as systime
from numba import jit

###############################################################################
#input/output
#These are just some helper functions to make make it easy to read/write hdf files. You don't have to use them
def hdf_write(data_file_name, field_names, fields):
    #Make data file to store all the data in. If it already exists, it might not overwright!
    if type(field_names) == str:
        field_names = [field_names]
        fields = [fields]
    
    data_file = h5py.File(data_file_name,'a')
    for idx, (field_name, field) in enumerate(zip(field_names, fields)):
        #This allows me to create or overwrite a dataset without me knowing if it is there or not
        try:
            if type(field) == ndarray:
                data_file.require_dataset(field_name, field.shape, field.dtype)
                dset = data_file[field_name]
                dset[...] = field
            else:
                data_file.require_dataset(field_name, shape(field), type(field))
                dset = data_file[field_name]
                dset[...] = field
        except TypeError:
            try:
                data_file.create_dataset(field_name, data = field)
            except RuntimeError:
                del data_file[field_name]
                data_file.create_dataset(field_name, data = field)
    data_file.close()

def hdf_read(data_file_name, field_names):
    #reads dataset specified in "field_name" (which are hdf paths to datasets)
    if type(field_names) == str:
        field_names = [field_names]
    return_arrays = []
    for field_name in field_names:
        data_file = h5py.File(data_file_name,'r')
        dataset = data_file[str(field_name)]
        data_array = dataset[...]
        return_arrays.append(data_array)
    data_file.close()
    if len(return_arrays) == 1:
        return return_arrays[0]
    else:
        return return_arrays

################################################################

@jit(nopython=True, cache=True)
def calc_overlap_function_jit(binned_x1, binned_x2, tau, bin_width, bins_tot):

    overlap_function = zeros(len(tau))
    binned_x2_temp = binned_x2.copy()
    length_tau = tau.shape[0]

    #Iterate over every value of tau
    for i in range(length_tau):

        tau_i = tau[i]
        tau_i_bins = tau_i/bin_width

        sum_val = 0
        x1_idx = 0
        x2_idx = 0
        length_binned_x1 = binned_x1.shape[0]
        length_binned_x2 = binned_x2_temp.shape[0]

        #Add the current tau to all the times of evebts at x2
        for j in range(length_binned_x2):
            binned_x2_temp[j] = binned_x2[j] + tau_i_bins
        
        while (x1_idx < length_binned_x1) and (x2_idx < length_binned_x2):
            #Get the time of an event in x1
            x1_current_bin_val = binned_x1[x1_idx]
            x1_idx += 1
            events_in_current_x1_bin = 1
            events_in_current_x2_bin_same_as_x1_bin = 0
            
            #Determine how many events in x1 have the SAME time (measured in bin number) (only happens if the bin width is greater than the holdoff time of the SCAs (0.5us))
            while (x1_idx < length_binned_x1) and (binned_x1[x1_idx] == x1_current_bin_val):
                events_in_current_x1_bin += 1
                x1_idx += 1
            
            #Find the first event in x2 that has the same time as the current x1 event (if it exists). Obviously I can stop checking if I see an event in x2 with time greater than the current x1 event.
            while (x2_idx < length_binned_x2) and (binned_x2_temp[x2_idx] < x1_current_bin_val):
                x2_idx += 1
            
            #Determine how many events in x2 have the SAME time (only happens if the bin width is greater than the holdoff time of the SCAs (0.5us))
            while (x2_idx < length_binned_x2) and (binned_x2_temp[x2_idx] == x1_current_bin_val):
                events_in_current_x2_bin_same_as_x1_bin += 1
                x2_idx += 1
            
            #The overlap integal for a given tau is determined from the sum of all products of matching times (from x1 and x2)
            sum_val =  sum_val + events_in_current_x1_bin*events_in_current_x2_bin_same_as_x1_bin

        #calculate overlap for current tau
        overlap_function[i] = 1/(bins_tot*bin_width)*sum_val*bin_width
    return overlap_function


def g2_calc(hdf_name):
    #Set parametes for the correlation
    bin_width = 50E-9
    tau_min = 700E-6
    tau_max = 800E-6
    # change the group name to whatever you want. Be sure to update the name in the plotting function
    processed_data_group_name = 'processed/50ns_bin/'

    hdf_file = h5py.File(hdf_name, 'a')
    total_entries = hdf_file['total_entries'][:][0]
    dset_records = hdf_file['records']
    times_all = dset_records['time'][:total_entries]*5E-9
    ch0 = dset_records['ch0'][:total_entries]
    ch1 = dset_records['ch1'][:total_entries]
    ch2 = dset_records['ch2'][:total_entries]
    ch3 = dset_records['ch3'][:total_entries]

    times_ch0 = times_all[ch0==1]
    times_ch1 = times_all[ch1==1]
    times_ch2 = times_all[ch2==1]
    times_ch3 = times_all[ch3==1]

    # Choose which channels you want to perform the g2 mesurement over
    # It is fine to make them both the same channel
    times_x1 = times_ch0
    times_x2 = times_ch1

    t_min = min([times_x1.min(), times_x2.min()])
    t_max = max([times_x1.max(), times_x2.max()])
    t_range = t_max - t_min

    binned_x1 = floor((times_x1 - t_min)/bin_width)
    binned_x2 = floor((times_x2 - t_min)/bin_width)

    bins_tot = ceil(t_range/bin_width)

    ave_I_x1 = len(binned_x1)/bins_tot
    ave_I_x2 = len(binned_x2)/bins_tot

    tau_range = tau_max - tau_min
    tau_element_n = round(tau_range/bin_width)
    tau = arange(tau_element_n)*bin_width + tau_min

    t0 = systime.time()
    print('Calculating overlap function...')
    overlap_function = calc_overlap_function_jit(binned_x1, binned_x2, tau, bin_width, bins_tot)

    print('Finished calculating overlap function...')
    t1 = systime.time()
    time_taken = t1-t0
    print('Calculation time = '+str(int(floor(time_taken/(60*60))))+'hrs '+str(int(floor(mod(time_taken, 60*60)/60)))+'mins '+str(int(mod(time_taken, 60)))+'secs')
    
    g2 = overlap_function/(ave_I_x1 * ave_I_x2)
    
    hdf_write(hdf_name, [processed_data_group_name + 'g2', processed_data_group_name + 'tau'], [g2, tau])


def plot_g2(hdf_name):
    # change the group name to the data that you want to plot
    processed_data_group_name = 'processed/50ns_bin/'

    [g2, tau] = hdf_read(hdf_name, [processed_data_group_name + 'g2', processed_data_group_name + 'tau'])
    bin_width = tau[1] - tau[0]
    
    fig = figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(tau*1E6, g2)
    ax1.set_xlim(tau.min()*1E6, tau.max()*1E6)
    ax1.set_xlabel(r'$\rm{\tau}$ $\rm{(\mu s)}$')
    ax1.set_ylabel(r'$\rm{g^{(2)}(\tau)}$')
    tight_layout()
    savefig('g2_{:.1f}ns_bins.pdf'.format(bin_width*1E9))
    show()


def histogram_of_deltas(hdf_name):
    # Set the desired bin width, and the min and max dt over which you want to plot
    bin_width = 50E-9
    dt_min = 900E-6
    dt_max = 1100E-6

    hdf_file = h5py.File(hdf_name, 'a')
    total_entries = hdf_file['total_entries'][:][0]
    dset_records = hdf_file['records']
    times_all = dset_records['time'][:total_entries]*5E-9
    ch0 = dset_records['ch0'][:total_entries]
    times_ch0 = times_all[ch0==1]

    deltas = diff(times_ch0)
    number_bins = int(round((dt_max - dt_min)/bin_width))
    histdata, edges = histogram(deltas, bins = number_bins, range=(dt_min, dt_max))
    centrepoints = (edges[:-1] + edges[1:])/2

    fig = figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(centrepoints*1E6, histdata)
    ax1.set_xlim(centrepoints.min()*1E6, centrepoints.max()*1E6)
    ax1.set_xlabel(r'$\rm{t_{n+1}-t_n}$ $\rm{(\mu s)}$')
    ax1.set_ylabel(r'$\rm{Occurrences}$')
    tight_layout()
    savefig('histogram_{:.1f}ns_bins.pdf'.format(bin_width*1E9))
    show()


###############################################################################
#Make program run now...
if __name__ == "__main__":

    hdf_name = 'pulse_record.hdf'

    # Show a simple histogram of the time between pulses
    histogram_of_deltas(hdf_name)

    # Calculate the g2 correlation function of 2 sets of pulses
    g2_calc(hdf_name)
    plot_g2(hdf_name)




































