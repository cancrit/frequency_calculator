import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import *
from matplotlib.path import Path
import scipy.interpolate as interpolate
import math
import matplotlib.gridspec as gridspec
import peakutils
from collections import OrderedDict
""" Author: Jacob Barfield
    email: jhbarfield@mail.roanoke.edu"""


class experimental_calculator:
	def __init__(self, params):
		plt.switch_backend('QT4Agg')
		#set up the variables that the program needs to function 
		self.files_list = params[0]
		self.buffer_x = int(params[1])
		self.buffer_y = int(params[2])
		self.threshold = float(params[3])
		self.threshold_percent = float(params[4])/100
		self.output_log_length = int(params[5])
		self.number_terminal_lines = int(params[6])
		self.min_baseline_order = int(params[7])
		self.max_baseline_order = int(params[8])
		self.baseline_order = self.min_baseline_order
		self.modes = ('automatic','horizontal line','custom line','remove noise','manual select')
		self.analysis_functions = {'horizontal line':self.horizontal_line_event_handling,
		                           'custom line': self.custom_line_event_handling,
		                           'remove noise': self.remove_noise_event_handling,
		                           'manual select':self.manual_event_handling,
		                           'automatic':self.automatic}
		self.keyboard_shortcuts = {'ctrl+a':self.append_periods,
		                           'ctrl+u':self.undo_append,
		                           'ctrl+r':self.redo_append,
		                           'ctrl+q':self.write_periods,
		                           'ctrl+Q':self.write_results,
		                           'ctrl+S':self.toggle_smooth,
		                           'ctrl+right':self.next_file,
		                           'ctrl+left':self.prev_file,
		                           'ctrl+up':self.next_cell,
		                           'ctrl+down':self.prev_cell,
		                           'ctrl+alt+s':self.toggle_both,
		                           'ctrl+l':self.load_periods,
		                           'left':self.decrease_baseline_order,
		                           'right':self.increase_baseline_order}
		self.mode = self.analysis_functions[self.modes[0]]
		self.files = []
		self.get_files()
		self.cell_ind = 0
		self.file_ind = 0
		self.log_ind = 0
		self.avg_period = None
		self.frequency = None
		self.std = None
		self.stdm = None
		self.periods_calculated = False
		self.cids = []
		self.path = []
		self.output_dict = OrderedDict()
		self.terminal_lines = []
		self.best_fit_line = None
		self.press = None
		self.output_dict_log = []
		self.get_data()
		self.smooth = False
		self.display_both = False
		self.plotted_points = np.array([[np.nan,np.nan]])
		self.started = False
		self.voi = None
		self.connectivity = None
		self.bursters = None
		self.seed = None
		#make the graphical user interface
		self.fig = plt.figure()
		
		self.grid = gridspec.GridSpec(5,6, height_ratios = [1,3,3,3,1], width_ratios = [2,1,1,1,1,1])
		self.grid2 = gridspec.GridSpecFromSubplotSpec(1,9, subplot_spec=self.grid[4,0:])

		self.ax1 = plt.subplot(self.grid[0,0:3])
		self.ax2 = plt.subplot(self.grid[0,3])
		self.ax3 = plt.subplot(self.grid[0,4])
		self.ax4 = plt.subplot(self.grid[0,5])
		self.ax5 = plt.subplot(self.grid[1,0])
		self.ax6 = plt.subplot(self.grid[2:3,0])
		self.ax7 = plt.subplot(self.grid[3,0])
		self.ax8 = plt.subplot(self.grid[1:4,1:])
		self.ax9 = plt.subplot(self.grid2[0,0])
		self.ax10 = plt.subplot(self.grid2[0,1])
		self.ax11 = plt.subplot(self.grid2[0,2])
		self.ax12 = plt.subplot(self.grid2[0,3])
		self.ax13 = plt.subplot(self.grid2[0,4])
		self.ax14 = plt.subplot(self.grid2[0,5])
		self.ax15 = plt.subplot(self.grid2[0,6])
		self.ax16 = plt.subplot(self.grid2[0,7])
		self.ax17 = plt.subplot(self.grid2[0,8])
		
		self.ax1.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='off',right='off',left='off',labelleft='off')		
		self.ax2.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='off',right='off',left='off',labelleft='off')		
		self.ax3.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='off',right='off',left='off',labelleft='off')
		self.ax4.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='off',right='off',left='off',labelleft='off')
		self.ax6.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='off',right='off',left='off',labelleft='off')
		self.ax7.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='off',right='off',left='off',labelleft='off')

		self.ax8.set_ylabel('potential (mV)', fontsize=12, fontweight='bold')
		self.ax8.set_xlabel('time (s)', fontsize=12, fontweight='bold')

		self.l1 = self.ax1.text(0.5, 0.5,'file data label 1',ha = 'left',va = 'center',fontsize = 14,fontweight = 'bold')
		self.l2 = self.ax2.text(0.5, 0.5,'file data label 2',ha = 'left',va = 'center',fontsize = 14,fontweight = 'bold')
		self.l3 = self.ax3.text(0.5, 0.5,'cell data label 1',ha = 'center',va = 'center',fontsize = 14,fontweight = 'bold')
		self.l4 = self.ax4.text(0.5, 0.5,'cell data label 2',ha = 'center',va = 'center',fontsize = 14,fontweight = 'bold')
		self.l5 = self.ax6.text(.05, .95,'output data label',va = 'top',fontsize = 14)		
		self.l6 = self.ax7.text(.05, .95,'terminal',va = 'top',multialignment= 'left',fontsize = 14)

		self.b1 = Button(self.ax9,'Undo Append')
		self.b2 = Button(self.ax10,'Redo Append')
		self.b3 = Button(self.ax11,'Write Periods')
		self.b4 = Button(self.ax12,'Write Results')
		self.b5 = Button(self.ax13,'Append Periods')
		self.b6 = Button(self.ax14,'Previous File')
		self.b7 = Button(self.ax15,'Previous')
		self.b8 = Button(self.ax16,'Next')
		self.b9 = Button(self.ax17,'Next File')		
		
		self.radio = RadioButtons(self.ax5, self.modes)

		self.b1.on_clicked(self.undo_append)
		self.b2.on_clicked(self.redo_append)
		self.b3.on_clicked(self.write_periods)
		self.b4.on_clicked(self.write_results)
		self.b5.on_clicked(self.append_periods)
		self.b6.on_clicked(self.prev_file)
		self.b7.on_clicked(self.prev_cell)
		self.b8.on_clicked(self.next_cell)
		self.b9.on_clicked(self.next_file)
		
		self.radio.on_clicked(self.change_mode)

		self.graph, = self.ax8.plot(self.coordinates[:,0],self.coordinates[:,1],color='blue')
		self.smoothed_graph, = self.ax8.plot(self.smoothed_coordinates[:,0],self.smoothed_coordinates[:,1],color='orange')
		self.plotted_peaks = self.ax8.scatter(self.plotted_points[:,0],self.plotted_points[:,1],color='black')
		self.smoothed_graph.set_visible(False)
		self.best_fit_line, = self.ax8.plot(0,0,'r-')
				
		#start up all the parts of the graphical user interface and the program itself
		self.adjust_title()
		self.adjust_limits()
		self.update_output_label()
		self.write_terminal()
		self.cid = self.graph.figure.canvas.mpl_connect('key_press_event', self.continueous_key_bindings)
		self.cid2 = self.graph.figure.canvas.mpl_connect('scroll_event', self.change_baseline_order)
		self.mode()
		mng = plt.get_current_fig_manager()
		mng.window.showMaximized()
		plt.tight_layout()
		plt.show()  

	def update_output_label(self):
		if self.periods_calculated == False:
			number_peaks = None
			output = ('output:\n' +
				      'avg period = ' + str(self.avg_period) + '\n' +
			          'frequency = ' + str(self.frequency) + '\n' +
			          'std = ' + str(self.std) + '\n' +
			          'stdm = ' + str(self.stdm) + '\n' +
			          'num peaks = ' + str(number_peaks))
		elif self.periods_calculated == True:
				number_peaks = len(self.periods) + 1
				output = 'output:\navg period = {:.4f}\nfrequency = {:.4f}\nstd = {:.4f}\nstdm = {:.4f}\nnum peaks = {:d}'.format(self.avg_period, 
					                                                                                                              self.frequency, 
					                                                                                                              self.std, 
					                                                                                                              self.stdm, 
					                                                                                                              number_peaks)
		self.l5.set_text(output)
		self.l5.set_multialignment('left')
		plt.draw()
	
	def write_terminal(self):
		line = 'terminal:\n'
		if len(self.terminal_lines) >= self.number_terminal_lines:
			for i in range(len(self.terminal_lines)-self.number_terminal_lines,len(self.terminal_lines)):
				line = line + '>>> {0}\n'.format(self.terminal_lines[i])
		elif len(self.terminal_lines) < self.number_terminal_lines:
			for i in range(len(self.terminal_lines)):
				line = line + '>>> {0}\n'.format(self.terminal_lines[i])
		self.l6.set_text(line)
		plt.draw()
	
	def adjust_limits(self):
		if self.smooth == True:
			type_coordinates = self.smoothed_coordinates
		elif self.smooth == False:
			type_coordinates = self.coordinates
		self.minx = np.min(type_coordinates[:,0])
		self.miny = np.min(type_coordinates[:,1])
		self.maxx = np.max(type_coordinates[:,0])
		self.maxy = np.max(type_coordinates[:,1])
		self.buffer_x = (self.maxx - self.minx)/20
		self.buffer_y = (self.maxy - self.miny)/20
		self.graph_minx = np.min(type_coordinates[:,0]) - self.buffer_x
		self.graph_miny = np.min(type_coordinates[:,1]) - self.buffer_y
		self.graph_maxx = np.max(type_coordinates[:,0]) + self.buffer_x
		self.graph_maxy = np.max(type_coordinates[:,1]) + self.buffer_y
		self.ax8.set_xlim(self.graph_minx,self.graph_maxx)
		self.ax8.set_ylim(self.graph_miny,self.graph_maxy)
		plt.draw()       
	
	def adjust_title(self):
		file = self.files[self.file_ind]
		cell_fraction = str(self.cell_ind + 1) + '/' + str(len(self.headers))
		file_fraction = str(self.file_ind + 1) + '/' + str(len(self.files))
		
		l1_text = file.split(self.base_location)[1]
		l2_text = 'file ' + file_fraction
		l3_text = self.headers[self.cell_ind]
		l4_text = 'cell ' + cell_fraction

		self.l1.set_text(l1_text)
		self.l2.set_text(l2_text)
		self.l3.set_text(l3_text)
		self.l4.set_text(l4_text)

		self.l1.set_ha('center')
		self.l1.set_va('center')
		self.l2.set_ha('center')
		self.l2.set_va('center')
		plt.draw()
	
	def next_cell(self, event):
		if self.cell_ind == self.data.shape[1] - 3:
			self.cell_ind = 0
		else:
			self.cell_ind += 1
		self.change_data('cell')
	
	def prev_cell(self, event):
		if self.cell_ind == 0:
			self.cell_ind = self.data.shape[1] - 3
		else:
			self.cell_ind -= 1
		self.change_data('cell')
	
	def next_file(self, event):
		if self.file_ind == len(self.files)-1:
			self.file_ind = 0
		else:
			self.file_ind += 1
		self.change_data('file')
	
	def prev_file(self, event):
		if self.file_ind == 0:
			self.file_ind = len(self.files)-1
		else:
			self.file_ind -= 1
		self.change_data('file')
			
	def change_data(self, type_change):
		if type_change == 'cell':
			self.reset_data()
		elif type_change == 'file':
			self.get_data(file_change = True, reset_output = True)
		self.adjust_title()
		self.update_output_label()
		self.adjust_limits()
		self.replot_data(reset_fit_line = True)
		self.started = False
		self.disconnect_connections()
		self.mode()
	
	def change_mode(self, label):
		self.disconnect_connections()
		if self.started == True:
			self.started = False
		self.replot_data(reset_fit_line=True)
		self.mode = self.analysis_functions[label]
		self.mode()
	
	def get_data(self, file_change = False, reset_output=False):
		if file_change == True:
			self.cell_ind = 0
		self.data = np.genfromtxt(self.files[self.file_ind],delimiter=',',skip_header=1,skip_footer=1)
		with open(self.files[self.file_ind],'r') as file:
			pulled_data = file.read().split('\n')[0]
		self.headers = [pulled_data.split(',')[index] for index in range(len(pulled_data.split(','))) if index >= 2]
		self.xdata = self.data[:,0]
		self.ydata = self.data[:,2+self.cell_ind]
		self.coordinates = np.column_stack((self.xdata,self.ydata))
		self.smoothed_coordinates = self.smooth_data()
		self.avg_dt = np.mean(self.xdata[1:]-self.xdata[0:-1])
		self.min_t = np.min(self.xdata)
		self.max_t = np.max(self.xdata)
		if reset_output == True:
			self.peaks = None
			self.periods = None
			self.avg_period = None
			self.frequency = None
			self.std = None
			self.stdm = None
			self.periods_calculated = False
			self.best_fit_polynomial = None
	
	def reset_data(self):
		self.xdata = self.data[:,0]
		self.ydata = self.data[:,2 + self.cell_ind]
		self.coordinates = np.column_stack((self.xdata,self.ydata))
		self.smoothed_coordinates = self.smooth_data()
		self.avg_dt = np.mean(self.xdata[1:]-self.xdata[0:-1])
		self.min_t = np.min(self.xdata)
		self.max_t = np.max(self.xdata)
		self.avg_period = None
		self.frequency = None
		self.std = None
		self.stdm = None
		self.periods_calculated = False
		self.best_fit_polynomial = None
	
	def replot_data(self,fit_line=False,reset_fit_line=False,reset_scatter=False,threshold=False,thresh_y=None,thresh_x=None,scatter=False,scatter_x=None,scatter_y=None):
		if self.smooth == False and self.display_both == False:
			self.graph.set_xdata(self.coordinates[:,0])
			self.graph.set_ydata(self.coordinates[:,1])
			self.smoothed_graph.set_visible(False)
			self.graph.set_visible(True)
		elif self.smooth == True and self.display_both == False:
			self.smoothed_graph.set_xdata(self.smoothed_coordinates[:,0])
			self.smoothed_graph.set_ydata(self.smoothed_coordinates[:,1])
			self.smoothed_graph.set_visible(True)
			self.graph.set_visible(False)
		elif self.display_both == True:
			self.graph.set_xdata(self.coordinates[:,0])
			self.graph.set_ydata(self.coordinates[:,1])
			self.smoothed_graph.set_xdata(self.smoothed_coordinates[:,0])
			self.smoothed_graph.set_ydata(self.smoothed_coordinates[:,1])
			self.smoothed_graph.set_visible(True)
			self.graph.set_visible(True)
		
		if fit_line == True:
			x_min = np.min(self.path[0])
			x_max = np.max(self.path[0])
			number_samples = (x_max - x_min)/self.avg_dt
			fit_line_xs = np.linspace(x_min,x_max,num=number_samples)
		
		if fit_line == True and self.best_fit_line == None:
			#create the best_fit_line
			self.best_fit_line, = self.ax8.plot(fit_line_xs,self.best_fit_polynomial(fit_line_xs),color = 'black')
		elif fit_line == True and self.best_fit_line != None:
			self.best_fit_line.set_xdata(fit_line_xs)
			self.best_fit_line.set_ydata(self.best_fit_polynomial(fit_line_xs))
		elif reset_fit_line == True:
			self.best_fit_line.set_xdata(0)
			self.best_fit_line.set_ydata(0)
		elif threshold == True:
			self.best_fit_line.set_xdata(thresh_x)
			self.best_fit_line.set_ydata(thresh_y)
		
		if scatter == True:
			self.plotted_peaks.remove()
			self.plotted_peaks = self.ax8.scatter(scatter_x,scatter_y,color='black')
		elif scatter == False and self.started == False:
			self.plotted_peaks.remove()
			self.plotted_peaks = self.ax8.scatter(np.nan,np.nan,color='black')
		
		if reset_scatter == True:
			self.plotted_peaks.remove()
			self.plotted_peaks = self.ax8.scatter(np.nan,np.nan,color='black')
		plt.draw()
	
	def horizontal_line_event_handling(self):
		self.cids = []
		cid = self.graph.figure.canvas.mpl_connect('button_press_event',self.horizontal_line)
		self.cids.append(cid)
	
	def custom_line_event_handling(self):
		self.cids = []
		cid1 = self.graph.figure.canvas.mpl_connect('button_press_event', self.custom_line_on_click)
		cid2 = self.graph.figure.canvas.mpl_connect('button_release_event', self.custom_line_on_release)
		cid3 = self.graph.figure.canvas.mpl_connect('motion_notify_event', self.custom_line_on_motion)
		self.cids.append(cid1)
		self.cids.append(cid2)
		self.cids.append(cid3)
	
	def remove_noise_event_handling(self):
		self.lasso = LassoSelector(self.ax8, self.lasso_on_select)
		self.cids = []
		cid = self.graph.figure.canvas.mpl_connect('key_press_event', self.remove_noise_key_press)
		self.cids.append(cid)
	
	def manual_event_handling(self):
		self.peaks = []
		self.cids = []
		cid = self.graph.figure.canvas.mpl_connect('button_press_event', self.manual_selection_on_click)
		cid2 = self.graph.figure.canvas.mpl_connect('key_press_event', self.manual_selection_on_key_press)
		self.cids.append(cid)
		self.cids.append(cid2)

	def automatic(self):
		if self.started == False:
			self.started = True
			type_coordinates = self.get_type_coordinates()
			self.initiate_automatic(type_coordinates)
		self.cids = []
		cid = self.graph.figure.canvas.mpl_connect('button_press_event',self.change_threshold_percent)
		cid2 = self.graph.figure.canvas.mpl_connect('key_press_event',self.change_threshold_percent_key_press)
		self.cids.append(cid)
		self.cids.append(cid2)

	def horizontal_line(self, event):
		if event.inaxes != self.ax8: return
		self.path = [[self.graph_minx,self.graph_maxx],[event.ydata,event.ydata]]
		self.best_fit_polynomial = interpolate.interp1d(self.path[0],self.path[1],kind='linear',bounds_error=False,fill_value='extrapolate')
		self.replot_data(fit_line=True)
		self.threshold_analysis()
	
	def custom_line_on_click(self, event):
		if event.inaxes != self.ax8:return
		if event.button == 1:
			self.press = True
			self.path = [[],[]]
			self.path[0].append(event.xdata)
			self.path[1].append(event.ydata)
		else:
			return
	
	def custom_line_on_motion(self, event):
		if self.press == None: return
		if event.inaxes != self.ax8: return
		previous_x = self.path[0][len(self.path[0])-1]
		self.path[0].append(event.xdata)
		self.path[1].append(event.ydata)
		self.best_fit_polynomial = interpolate.interp1d(self.path[0],self.path[1],kind='linear',bounds_error=False,fill_value='extrapolate')
		self.replot_data(fit_line=True)
	
	def custom_line_on_release(self, event):
		if event.button == 1:
			if self.press == None:
				return
			elif self.press == True:
				self.press = None
				self.best_fit_polynomial = interpolate.interp1d(self.path[0],self.path[1],kind='linear',bounds_error=False,fill_value='extrapolate')
				self.replot_data(fit_line=True)
				self.threshold_analysis()
			self.disconnect_connections()
		else:
			return
	
	def remove_noise_key_press(self, event):
		if event.key == 'enter':
			if self.smooth == True and self.display_both == False:
				self.find_peaks(self.smoothed_coordinates)
			elif self.smooth == False and self.display_both == False:
				self.find_peaks(self.coordinates)
			elif self.display_both == True:
				self.find_peaks(self.coordinates)
			self.peaks = np.array(self.peaks)
			self.replot_data()
			self.disconnect_connections()
			self.peak_analysis()

	def find_peaks(self, type_coordinates):
		slice1 = type_coordinates[0:-2,0]
		slice2 = type_coordinates[1:-1,0]
		slice3 = type_coordinates[2:,0]
		spike_starts = np.where(np.isnan(slice1) & np.isfinite(slice2) & np.isfinite(slice3))[0] + 1
		spike_ends = np.where(np.isfinite(slice1) & np.isfinite(slice2) & np.isnan(slice3))[0] + 1
		peak_indexes = np.where(np.isnan(slice1) & np.isfinite(slice2) & np.isnan(slice3))[0] + 1
		#determine the type of system that is present.
		self.peaks = None
		peaks_started = False
		if len(spike_starts) != len(spike_ends):
			if math.fabs(len(spike_ends)-len(spike_starts)) <= 1:
				#there are different numbers of spike starts and spike ends.
				if spike_ends[0] < spike_starts[0]: 
					#a spike was cut up in the beginning of the graph
					for index in range(len(spike_ends)):
						if index == 0:
							local_max = self.find_max(0,spike_ends[0])
						elif index != 0:
							local_max = self.find_max(spike_starts[index-1], spike_ends[index])
						if peaks_started == False:
							self.peaks = local_max
							peaks_started = True
						elif peaks_started == True:
							self.peaks = np.row_stack((self.peaks,local_max))
				elif spike_starts[len(spike_starts)-1] > spike_ends[len(spike_ends)-1]:
					#a spike was cut up at the end of the graph
					for index in range(len(spike_starts)):
						if index != len(spike_starts)-1:
							local_max = self.find_max(spike_starts[index], spike_ends[index])
						elif index == len(spike_starts)-1:
							local_max = self.find_max(spike_starts[index],type_coordinates.shape[0]-1)
						if peaks_started == False:
							self.peaks = local_max
							peaks_started = True
						elif peaks_started == True:
							self.peaks = np.row_stack((self.peaks,local_max))
				else:
					print('ERROR IN LENGTH OF THE SPIKE START AND SPIKE END LISTS')
			else:
				print('ERROR IN THE LENGTH OF THE SPIKE START AND SPIKE END LISTS')
				print('len(spike_starts) = ',len(spike_starts))
				print('len(spike_ends) = ',len(spike_ends))
		elif len(spike_starts) == len(spike_ends):
			#there are the same number of spike starts and spike ends.
			if spike_ends[0] < spike_starts[0]:
				# the system has spikes that are cut at the beginning and end of the graph
				for index in range(len(spike_ends)):
					if index == 0:
						local_max = self.find_max(0,spike_ends[0])
					elif index > 0 and index != len(spike_ends)-1:
						local_max = self.find_max(spike_starts[index-1], spike_ends[index])
					elif index > 0 and index == len(spike_ends)-1:
						local_max = self.find_max(spike_starts[index],type_coordinates.shape[0]-1)
					if peaks_started == False:
						self.peaks = local_max
						peaks_started = True
					elif peaks_started == True:
						self.peaks = np.row_stack((self.peaks,local_max))
			elif spike_starts[0] < spike_ends[0]:
				#the system is normal
				for index in range(len(spike_starts)):
					local_max = self.find_max(spike_starts[index], spike_ends[index])
					if peaks_started == False:
						self.peaks = local_max
						peaks_started = True
					elif peaks_started == True:
						self.peaks = np.row_stack((self.peaks,local_max))
			else:
				print('ERROR IN THE POSITIONING OF THE SPIKES IN THE SYSTEM')
	
	def manual_selection_on_click(self, event):
		if self.smooth == True:
			dx = self.smoothed_coordinates[:,0] - event.xdata
			dy = self.smoothed_coordinates[:,1] - event.ydata
			dx2 = dx * dx
			dy2 = dy * dy
			distances = np.sqrt(dx2 + dy2)
			peak = self.smoothed_coordinates[distances[:] == np.min(distances[:])][0]
			self.peaks.append(peak)
		elif self.smooth == False:
			dx = self.coordinates[:,0] - event.xdata
			dy = self.coordinates[:,1] - event.ydata
			dx2 = dx * dx
			dy2 = dy * dy
			distances = np.sqrt(dx2 + dy2)
			peak = self.coordinates[distances[:] == np.min(distances[:])][0]
			self.peaks.append(peak)
	
	def manual_selection_on_key_press(self, event):
		if event.key == 'enter':
			self.peaks = np.array(self.peaks)
			self.replot_data()
			self.peak_analysis()
			self.disconnect_connections()

	def change_threshold_percent(self, event):
		if event.inaxes != self.ax8: return
		if event.button == 1:
			y = (event.ydata - self.baseline_function(event.xdata)) / (self.max_y - self.min_y)
			if y >= self.min_y and y <= self.max_y:
				self.threshold_percent = y
				if self.smooth == True:
					type_coordinates = self.smoothed_coordinates
				elif self.smooth == False:
					type_coordinates = self.coordinates
				self.automatic_analysis(type_coordinates)

	def change_threshold_percent_key_press(self, event):
		if (event.key == 'up') or (event.key == 'down') or (event.key == 'alt+up') or (event.key == 'alt+down'):
			if event.key == 'up':
				self.threshold_percent += .01
			elif event.key == 'down':
				self.threshold_percent -= .01
			elif event.key == 'alt+up':
				self.threshold_percent += .1
			elif event.key == 'alt+down':
				self.threshold_percent -= .1
			if self.smooth == True:
				type_coordinates = self.smoothed_coordinates
			elif self.smooth == False:
				type_coordinates = self.coordinates
			self.automatic_analysis(type_coordinates)
		else:
			return
	
	def find_max(self, index1, index2):
		region = self.coordinates[index1:index2+1,1]
		local_max = self.coordinates[index1:index2+1,:][region == np.max(region)]
		return local_max
	
	def lasso_on_select(self, verts):
		path = Path(verts)
		if self.smooth == True and self.display_both == False:
			mask = np.array([path.contains_point(xy) for xy in self.smoothed_coordinates])
			pos = np.where(mask)[0]
			self.smoothed_coordinates[pos,0] = np.nan
			self.smoothed_coordinates[pos,1] = np.nan
			#self.coordinates = self.coordinates[mask,:]
			self.smoothed_graph.set_xdata(self.smoothed_coordinates[:,0])
			self.smoothed_graph.set_ydata(self.smoothed_coordinates[:,1])
		elif self.smooth == False and self.display_both == False:
			mask = np.array([path.contains_point(xy) for xy in self.coordinates])
			pos = np.where(mask)[0]
			self.coordinates[pos,0] = np.nan
			self.coordinates[pos,1] = np.nan
			#self.coordinates = self.coordinates[mask,:]
			self.graph.set_xdata(self.coordinates[:,0])
			self.graph.set_ydata(self.coordinates[:,1])
		elif self.display_both == True:
			mask = np.array([path.contains_point(xy) for xy in self.coordinates])
			pos = np.where(mask)[0]
			self.coordinates[pos,0] = np.nan
			self.coordinates[pos,1] = np.nan
			#self.coordinates = self.coordinates[mask,:]
			self.graph.set_xdata(self.coordinates[:,0])
			self.graph.set_ydata(self.coordinates[:,1])
			mask = np.array([path.contains_point(xy) for xy in self.smoothed_coordinates])
			pos = np.where(mask)[0]
			self.smoothed_coordinates[pos,0] = np.nan
			self.smoothed_coordinates[pos,1] = np.nan
			#self.coordinates = self.coordinates[mask,:]
			self.smoothed_graph.set_xdata(self.smoothed_coordinates[:,0])
			self.smoothed_graph.set_ydata(self.smoothed_coordinates[:,1])
		plt.draw()
	
	def threshold_analysis(self):
		zero_points = self.best_fit_polynomial(self.coordinates[:,0])
		current_points = self.coordinates[1:,:]
		last_points = self.coordinates[0:-1,:]
		slope = (current_points[0:,1] - last_points[0:,1]) / (current_points[0:,0] - last_points[0:,0])
		smoothed_slope,cross = self.get_slope(current_points[:,0], last_points[:,0],zero_points[1:])
		#self.peaks = self.coordinates[1:][(current_points[0:,1] >= zero_points[1:]) & (last_points[0:,1] <= zero_points[1:]) & (smoothed_slope >= 0) & (slope >= 0)]
		self.peaks = self.coordinates[1:][(smoothed_slope >= 0) & (cross == True)]
		self.peak_analysis()
	
	def get_derivative(self):
		data = self.smooth_data()
		self.derivative = interpolate.interp1d(data[:,0],data[:,1],kind='linear',bounds_error=False,fill_value='extrapolate')
	
	def get_slope(self, tf, ti, zero):
		self.get_derivative()
		yi = self.derivative(ti)
		yf = self.derivative(tf)
		dy = yf - yi
		dx = tf - ti
		slope = np.divide(dy,dx)
		cross = (yi <= zero) & (yf >= zero)
		return (slope,cross)
	
	def smooth_data(self):
		slice1 = self.coordinates[0:-4,:]
		slice2 = self.coordinates[1:-3,:]
		slice3 = self.coordinates[2:-2,:]
		slice4 = self.coordinates[3:-1,:]
		slice5 = self.coordinates[4:,:]
		new_data = slice1 + slice2 + slice3 + slice4 + slice5
		new_data = new_data * (1/5)
		return new_data
	
	def peak_analysis(self):
		if self.peaks.shape[0] > 1:
			self.periods = self.peaks[1:,0] - self.peaks[0:-1,0]
			peaks = self.peaks[1:][self.peaks[1:,0] - self.peaks[0:-1,0] > self.threshold]
			peaks = np.concatenate((peaks,self.peaks[0:-1][self.peaks[1:,0] - self.peaks[0:-1,0] > self.threshold]))
			self.periods = self.periods[self.periods > self.threshold]
			if self.periods.shape[0] > 1:
				self.periods_calculated = True
				self.avg_period = np.mean(self.periods)
				self.frequency = 1/self.avg_period
				self.std = np.std(self.periods)
				self.stdm = self.std / math.sqrt(len(self.periods))
				self.replot_data(scatter=True,scatter_x=peaks[:,0],scatter_y=peaks[:,1])
				self.update_output_label()
				self.disconnect_connections()
				self.mode()
				self.add_terminal_line('analysis complete')
			elif self.periods.shape[0] == 1 and self.peaks.shape[0] == 2:
				self.periods_calculated = True
				self.avg_period = np.mean(self.periods)
				self.frequency = 1/self.avg_period
				self.std = np.std(self.periods)
				self.stdm = self.std / math.sqrt(len(self.periods))
				self.replot_data(scatter = True, scatter_x = peaks[:,0], scatter_y = peaks[:,1])
				self.update_output_label()
				self.disconnect_connections()
				self.mode()
				self.add_terminal_line('analysis complete')
			else:
				self.replot_data(reset_scatter= True)
	
	def disconnect_connections(self):
		if len(self.cids) >= 1 and self.cids != []:
			for index in range(len(self.cids)):
				self.graph.figure.canvas.mpl_disconnect(self.cids[index])
			self.cids = []
		if self.mode == self.analysis_functions['remove noise']:
			self.lasso.disconnect_events()
	
	def continueous_key_bindings(self, event):
		if event.key in self.keyboard_shortcuts:
			path = self.files[self.file_ind].split(self.base_location)[1].split('/')
			self.drug = path[0]
			self.concentration = path[len(path)-1].split('mM')[0].strip()
			key_func = self.keyboard_shortcuts[event.key]
			key_func(event)
			self.replot_data()
	
	def get_files(self):
		with open(self.files_list, 'r') as file:
			pulled_data = file.read().split('\n')
			for index in range(len(pulled_data)):
				if index >= 1:
					self.files.append(pulled_data[index])
				elif index == 0:
					self.base_location = '{}/'.format(pulled_data[index])
		self.output_file_name = 'output.csv'.format(self.base_location)
		self.period_output_file_name = 'period_output.csv'.format(self.base_location)

	def undo_append(self, event):
		status = False
		if self.log_ind != 0:
			status = True
			self.log_ind -= 1
		self.output_dict = self.output_dict_log[self.log_ind]
		if status == True:
			self.add_terminal_line('undo append')

	def redo_append(self, event):
		status = False
		if self.log_ind != len(self.output_dict_log) - 1:
			status = True
			self.log_ind += 1
		self.output_dict = self.output_dict_log[self.log_ind]
		if status == True:
			self.add_terminal_line('redo append')

	def append_periods(self, event):
		path = self.files[self.file_ind].split(self.base_location)[1].split('/')
		self.drug = path[0]
		self.concentration = path[len(path)-1].split('mM')[0].strip()
		if self.drug not in self.output_dict:
			self.output_dict[self.drug] = OrderedDict()
		if self.concentration not in self.output_dict[self.drug]:
			self.output_dict[self.drug][self.concentration] = self.periods
		elif self.concentration in self.output_dict[self.drug]:
			period_list = self.output_dict[self.drug][self.concentration]
			period_list = np.concatenate((period_list, self.periods))
			self.output_dict[self.drug][self.concentration] = period_list
		if self.log_ind != len(self.output_dict_log)-1:
			dicts_to_remove = []
			for index in range(self.log_ind+1,len(self.output_dict_log)):
				dicts_to_remove.append(self.output_dict_log[index])
			for output_dictionary in dicts_to_remove:
				self.output_dict_log.remove(output_dictionary)
		self.output_dict_log.append(self.output_dict)
		if len(self.output_dict_log) > self.output_log_length:
			self.output_dict_log.remove(self.output_dict_log[0])
		self.log_ind = len(self.output_dict_log) - 1
		self.add_terminal_line('output appended')

	def write_periods(self, event):
		path = self.files[self.file_ind].split(self.base_location)[1].split('/')
		self.drug = path[0]
		self.concentration = path[len(path)-1].split('mM')[0].strip()
		with open(self.period_output_file_name,'w') as file:
			lines = []
			for drug in self.output_dict:
				for concentration in self.output_dict[drug]:
					periods = self.output_dict[drug][concentration]
					line = drug + ',' + concentration
					for period in range(len(periods)):
						line = line +  ',' + str(periods[period])
					lines.append(line)
			for index in range(len(lines)):
				if index != len(lines)-1:
					lines[index] = lines[index] + '\n'
				file.write(lines[index])
		self.add_terminal_line('periods recorded')

	def write_results(self, event):
		path = self.files[self.file_ind].split(self.base_location)[1].split('/')
		self.drug = path[0]
		self.concentration = path[len(path)-1].split('mM')[0].strip()
		with open(self.output_file_name,'w') as file:
			lines = []
			for drug in self.output_dict:
				for concentration in self.output_dict[drug]:
					periods = self.output_dict[drug][concentration]
					avg_period = np.mean(periods)
					frequency = 1/avg_period
					std = np.std(periods)
					stdm = std / math.sqrt(len(periods))
					line = drug + ',' + concentration + ',' + str(avg_period) + ',' + str(frequency) + ',' + str(std) + ',' + str(stdm)
					lines.append(line)
			for index in range(len(lines)):
				if index != len(lines)-1:
					lines[index] = lines[index] + '\n'
				file.write(lines[index])
		self.add_terminal_line('results recorded')

	def toggle_smooth(self, event):
		if self.smooth == True:
			self.smooth = False
			line = 'reduced noise: off'
		elif self.smooth == False:
			self.smooth = True
			line = 'reduced noise: on'
		self.check_automatic()
		self.add_terminal_line(line)

	def check_automatic(self):
		if self.started == True:
			type_coordinates = self.get_type_coordinates()
			self.initiate_automatic(type_coordinates)

	def toggle_both(self, event):
		if self.display_both == True:
			self.display_both = False
			line = 'display both: off'
		elif self.display_both == False:
			self.display_both = True
			line = 'display both: on'
		self.add_terminal_line(line)

	def automatic_analysis(self, type_coordinates):
		fixed_coordinates = np.copy(self.fixed_coordinates)
		threshold = self.min_y + ((self.max_y - self.min_y)*self.threshold_percent)
		fixed_coordinates[:,1][self.fixed_coordinates[:,1] <= threshold] = 0
		thresh_c = np.copy(fixed_coordinates)
		thresh_c[:,1] = threshold + self.baseline
		indexes = peakutils.indexes(fixed_coordinates[:,1], thres=0, min_dist=self.threshold/self.avg_dt)
		self.peaks = None
		self.peaks = type_coordinates[indexes,:]
		self.replot_data(threshold=True,thresh_x=thresh_c[:,0],thresh_y=thresh_c[:,1])
		self.peak_analysis()

	def change_baseline_order(self, event):
		self.baseline_order = self.baseline_order + event.step
		if self.baseline_order < self.min_baseline_order:
			self.baseline_order = self.min_baseline_order
		self.add_terminal_line('baseline order = {}'.format(self.baseline_order))
		type_coordinates = self.get_type_coordinates()
		self.initiate_automatic(type_coordinates)

	def increase_baseline_order(self, event):
		self.baseline_order = self.baseline_order + 1
		if self.baseline_order < self.min_baseline_order:
			self.baseline_order = self.min_baseline_order
		elif self.baseline_order > self.max_baseline_order:
			self.baseline_order = self.max_baseline_order
		self.add_terminal_line('baseline order = {}'.format(self.baseline_order))
		type_coordinates = self.get_type_coordinates()
		self.initiate_automatic(type_coordinates) 

	def decrease_baseline_order(self, event):
		self.baseline_order = self.baseline_order - 1
		if self.baseline_order < 0:
			self.baseline_order = 0
		elif self.baseline_order > self.max_baseline_order:
			self.baseline_order = self.max_baseline_order
		self.add_terminal_line('baseline order = {}'.format(self.baseline_order))
		type_coordinates = self.get_type_coordinates()
		self.initiate_automatic(type_coordinates) 

	def add_terminal_line(self, line):
		self.terminal_lines.append(line)
		if len(self.terminal_lines) == self.number_terminal_lines + 1:
			self.terminal_lines.remove(self.terminal_lines[0])
		self.write_terminal()

	def initiate_automatic(self, type_coordinates):
		self.started = True
		self.baseline = peakutils.baseline(type_coordinates[:,1],self.baseline_order)
		baseline_c = np.copy(type_coordinates)
		baseline_c[:,1] = self.baseline
		self.baseline_function = interpolate.interp1d(baseline_c[:,0],baseline_c[:,1],kind='linear',bounds_error=False,fill_value='extrapolate')
		self.fixed_coordinates = np.copy(type_coordinates)
		self.fixed_coordinates[:,1] = self.fixed_coordinates[:,1] - self.baseline
		self.min_y = np.min(self.fixed_coordinates[:,1])
		self.max_y = np.max(self.fixed_coordinates[:,1])
		self.automatic_analysis(type_coordinates)

	def get_type_coordinates(self):
		if self.smooth == True:
			return self.smoothed_coordinates
		elif self.smooth == False:
			return self.coordinates

	def load_periods(self, event):
		with open(self.period_output_file_name,'r') as file:
			pulled_data = file.read().split('\n')
		for index in range(len(pulled_data)):
			pulled_data[index] = pulled_data[index].split(',')
			drug = pulled_data[index][0]
			concentration = pulled_data[index][1]
			periods = np.array([float(pulled_data[index][x]) for x in range(2,len(pulled_data[index]))])
			if drug not in self.output_dict:
				self.output_dict[drug] = OrderedDict()
			if concentration not in self.output_dict[drug]:
				self.output_dict[drug][concentration] = periods
			elif concentration in self.output_dict[drug]:
				period_list = self.output_dict[drug][concentration]
				period_list = np.concatenate((period_list,periods))
		self.add_terminal_line('periods loaded')

class computational_calculator:
	def __init__(self, params):
		plt.switch_backend('QT4Agg')
		#set up the variables that the program needs to function 
		self.files_list = params[0]
		self.buffer_x = int(params[1])
		self.buffer_y = int(params[2])
		self.threshold = int(params[3])*1000
		self.threshold_percent = float(params[4])/100
		self.output_log_length = int(params[5])
		self.number_terminal_lines = int(params[6])
		self.min_baseline_order = int(params[7])
		self.max_baseline_order = int(params[8])
		self.baseline_order = self.min_baseline_order
		self.modes = ('automatic','horizontal line','custom line','remove noise','manual select')
		self.analysis_functions = {'horizontal line':self.horizontal_line_event_handling,
		                           'custom line': self.custom_line_event_handling,
		                           'remove noise': self.remove_noise_event_handling,
		                           'manual select':self.manual_event_handling,
		                           'automatic':self.automatic}
		self.keyboard_shortcuts = {'ctrl+a':self.append_periods,
		                           'ctrl+u':self.undo_append,
		                           'ctrl+r':self.redo_append,
		                           'ctrl+q':self.write_periods,
		                           'ctrl+Q':self.write_results,
		                           'ctrl+S':self.toggle_smooth,
		                           'ctrl+right':self.next_file,
		                           'ctrl+left':self.prev_file,
		                           'ctrl+up':self.next_cell,
		                           'ctrl+down':self.prev_cell,
		                           'ctrl+alt+s':self.toggle_both,
		                           'ctrl+l':self.load_periods,
		                           'left':self.decrease_baseline_order,
		                           'right':self.increase_baseline_order}
		self.mode = self.analysis_functions[self.modes[0]]
		self.files = []
		self.var = None
		self.voi = 0.00
		self.bursters = 0.00
		self.connectivity = 0.00
		self.seed = 0.00
		self.concentration = 0.00
		self.get_files()
		self.cell_ind = 0
		self.file_ind = 0
		self.log_ind = 0
		self.avg_period = None
		self.frequency = None
		self.std = None
		self.stdm = None
		self.periods_calculated = False
		self.cids = []
		self.path = []
		self.output_dict = OrderedDict()
		self.terminal_lines = []
		self.best_fit_line = None
		self.press = None
		self.output_dict_log = []
		self.get_data(file_change=True)
		self.smooth = False
		self.display_both = False
		self.plotted_points = np.array([[np.nan,np.nan]])
		self.started = False
		#make the graphical user interface
		self.fig = plt.figure()
		
		self.grid = gridspec.GridSpec(5,6, height_ratios = [1,3,3,3,1], width_ratios = [2,1,1,1,1,1])
		self.grid2 = gridspec.GridSpecFromSubplotSpec(1,9, subplot_spec=self.grid[4,0:])
		self.grid3 = gridspec.GridSpecFromSubplotSpec(1,10, subplot_spec=self.grid[0,0:])

		self.ax1 = plt.subplot(self.grid3[0,0:2])
		self.ax2 = plt.subplot(self.grid3[0,2:4])
		self.ax3 = plt.subplot(self.grid3[0,4:6])
		self.ax4 = plt.subplot(self.grid3[0,6:8])
		self.ax5 = plt.subplot(self.grid3[0,8:])
		self.ax6 = plt.subplot(self.grid[1,0])
		self.ax7 = plt.subplot(self.grid[2:3,0])
		self.ax8 = plt.subplot(self.grid[3,0])
		self.ax9 = plt.subplot(self.grid[1:4,1:])
		self.ax10 = plt.subplot(self.grid2[0,0])
		self.ax11 = plt.subplot(self.grid2[0,1])
		self.ax12 = plt.subplot(self.grid2[0,2])
		self.ax13 = plt.subplot(self.grid2[0,3])
		self.ax14 = plt.subplot(self.grid2[0,4])
		self.ax15 = plt.subplot(self.grid2[0,5])
		self.ax16 = plt.subplot(self.grid2[0,6])
		self.ax17 = plt.subplot(self.grid2[0,7])
		self.ax18 = plt.subplot(self.grid2[0,8])
		
		self.ax1.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='off',right='off',left='off',labelleft='off')		
		self.ax2.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='off',right='off',left='off',labelleft='off')		
		self.ax3.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='off',right='off',left='off',labelleft='off')
		self.ax4.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='off',right='off',left='off',labelleft='off')
		self.ax5.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='off',right='off',left='off',labelleft='off')
		self.ax7.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='off',right='off',left='off',labelleft='off')
		self.ax8.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='off',right='off',left='off',labelleft='off')

		self.ax9.set_ylabel('potential (mV)', fontsize=12, fontweight='bold')
		self.ax9.set_xlabel('time (s)', fontsize=12, fontweight='bold')

		self.l1 = self.ax1.text(0.5, 0.5,'label 1',ha = 'center',va = 'center',fontsize = 14,fontweight = 'bold')
		self.l2 = self.ax2.text(0.5, 0.5,'label 2',ha = 'center',va = 'center',fontsize = 14,fontweight = 'bold')
		self.l3 = self.ax3.text(0.5, 0.5,'label 3',ha = 'center',va = 'center',fontsize = 14,fontweight = 'bold')
		self.l4 = self.ax4.text(0.5, 0.5,'label 4',ha = 'center',va = 'center',fontsize = 14,fontweight = 'bold')
		self.l5 = self.ax5.text(0.5, 0.5,'label 5',ha = 'center',va = 'center',fontsize = 14,fontweight = 'bold')
		self.l6 = self.ax7.text(.05, .95,'label 6',va = 'top',fontsize = 14)		
		self.l7 = self.ax8.text(.05, .95,'label 7',va = 'top',multialignment= 'left',fontsize = 14)

		self.b1 = Button(self.ax10,'Undo Append')
		self.b2 = Button(self.ax11,'Redo Append')
		self.b3 = Button(self.ax12,'Write Periods')
		self.b4 = Button(self.ax13,'Write Results')
		self.b5 = Button(self.ax14,'Append Periods')
		self.b6 = Button(self.ax15,'Previous File')
		self.b7 = Button(self.ax16,'Previous')
		self.b8 = Button(self.ax17,'Next')
		self.b9 = Button(self.ax18,'Next File')		
		
		self.radio = RadioButtons(self.ax6, self.modes)

		self.b1.on_clicked(self.undo_append)
		self.b2.on_clicked(self.redo_append)
		self.b3.on_clicked(self.write_periods)
		self.b4.on_clicked(self.write_results)
		self.b5.on_clicked(self.append_periods)
		self.b6.on_clicked(self.prev_file)
		self.b7.on_clicked(self.prev_cell)
		self.b8.on_clicked(self.next_cell)
		self.b9.on_clicked(self.next_file)
		
		self.radio.on_clicked(self.change_mode)

		self.graph, = self.ax9.plot(self.coordinates[:,0],self.coordinates[:,1],color='blue')
		self.smoothed_graph, = self.ax9.plot(self.smoothed_coordinates[:,0],self.smoothed_coordinates[:,1],color='orange')
		self.plotted_peaks = self.ax9.scatter(self.plotted_points[:,0],self.plotted_points[:,1],color='black')
		self.smoothed_graph.set_visible(False)
		self.best_fit_line, = self.ax9.plot(0,0,'r-')
				
		#start up all the parts of the graphical user interface and the program itself
		self.adjust_title()
		self.adjust_limits()
		self.update_output_label()
		self.write_terminal()
		self.cid = self.graph.figure.canvas.mpl_connect('key_press_event', self.continueous_key_bindings)
		self.cid2 = self.graph.figure.canvas.mpl_connect('scroll_event', self.change_baseline_order)
		self.mode()
		mng = plt.get_current_fig_manager()
		mng.window.showMaximized()
		plt.tight_layout()
		plt.show()  

	def update_output_label(self):
		if self.periods_calculated == False:
			number_peaks = None
			output = ('output:\n' +
				      'avg period = ' + str(self.avg_period) + '\n' +
			          'frequency = ' + str(self.frequency) + '\n' +
			          'std = ' + str(self.std) + '\n' +
			          'stdm = ' + str(self.stdm) + '\n' +
			          'num peaks = ' + str(number_peaks))
		elif self.periods_calculated == True:
				number_peaks = len(self.periods) + 1
				output = 'output:\navg period = {:.4f}\nfrequency = {:.4f}\nstd = {:.4f}\nstdm = {:.4f}\nnum peaks = {:d}'.format(self.avg_period, 
					                                                                                                              self.frequency, 
					                                                                                                              self.std, 
					                                                                                                              self.stdm, 
					                                                                                                              number_peaks)
		self.l6.set_text(output)
		self.l6.set_multialignment('left')
		plt.draw()
	
	def write_terminal(self):
		line = 'terminal:\n'
		if len(self.terminal_lines) >= self.number_terminal_lines:
			for i in range(len(self.terminal_lines)-self.number_terminal_lines,len(self.terminal_lines)):
				line = line + '>>> {0}\n'.format(self.terminal_lines[i])
		elif len(self.terminal_lines) < self.number_terminal_lines:
			for i in range(len(self.terminal_lines)):
				line = line + '>>> {0}\n'.format(self.terminal_lines[i])
		self.l7.set_text(line)
		plt.draw()
	
	def adjust_limits(self):
		if self.smooth == True:
			type_coordinates = self.smoothed_coordinates
		elif self.smooth == False:
			type_coordinates = self.coordinates
		self.minx = np.min(type_coordinates[:,0])
		self.miny = np.min(type_coordinates[:,1])
		self.maxx = np.max(type_coordinates[:,0])
		self.maxy = np.max(type_coordinates[:,1])
		self.buffer_x = (self.maxx - self.minx)/20
		self.buffer_y = (self.maxy - self.miny)/20
		self.graph_minx = np.min(type_coordinates[:,0]) - self.buffer_x
		self.graph_miny = np.min(type_coordinates[:,1]) - self.buffer_y
		self.graph_maxx = np.max(type_coordinates[:,0]) + self.buffer_x
		self.graph_maxy = np.max(type_coordinates[:,1]) + self.buffer_y
		self.ax9.set_xlim(self.graph_minx,self.graph_maxx)
		self.ax9.set_ylim(self.graph_miny,self.graph_maxy)
		plt.draw()       
	
	def adjust_title(self):
		l1_text = '{} = {}'.format(self.var,self.voi)
		l2_text = 'concentration = {}'.format(self.concentration)
		l3_text = 'connectivity = {}'.format(self.connectivity)
		l4_text = 'bursters = {}'.format(self.bursters)
		l5_text = 'seed = {}'.format(self.seed)

		self.l1.set_text(l1_text)
		self.l2.set_text(l2_text)
		self.l3.set_text(l3_text)
		self.l4.set_text(l4_text)
		self.l5.set_text(l5_text)

		#self.l1.set_ha('center')
		#self.l1.set_va('center')
		#self.l2.set_ha('center')
		#self.l2.set_va('center')
		plt.draw()
	
	def next_cell(self, event):
		if self.cell_ind == self.data.shape[1] - 2:
			self.cell_ind = 0
		else:
			self.cell_ind += 1
		self.change_data('cell')
	
	def prev_cell(self, event):
		if self.cell_ind == 0:
			self.cell_ind = self.data.shape[1] - 2
		else:
			self.cell_ind -= 1
		self.change_data('cell')
	
	def next_file(self, event):
		if self.file_ind == len(self.files)-1:
			self.file_ind = 0
		else:
			self.file_ind += 1
		self.change_data('file')
	
	def prev_file(self, event):
		if self.file_ind == 0:
			self.file_ind = len(self.files)-1
		else:
			self.file_ind -= 1
		self.change_data('file')
			
	def change_data(self, type_change):
		if type_change == 'cell':
			self.reset_data()
		elif type_change == 'file':
			self.get_data(file_change = True, reset_output = True)
		self.seed = self.headers[self.cell_ind].split(' ')[1]
		self.adjust_title()
		self.update_output_label()
		self.adjust_limits()
		self.replot_data(reset_fit_line = True)
		self.started = False
		self.disconnect_connections()
		self.mode()
	
	def change_mode(self, label):
		self.disconnect_connections()
		if self.started == True:
			self.started = False
		self.replot_data(reset_fit_line=True)
		self.mode = self.analysis_functions[label]
		self.mode()
	
	def get_data(self, file_change = False, reset_output=False):
		if file_change == True:
			self.cell_ind = 0
			# self.path = self.files[self.file_ind].split(self.base_location)[1].split('/')
			# for folder in range(len(self.path)):
			# 	self.path[folder] = self.path[folder].split('_')
			# self.voi = self.path[0][1]
			# self.var = self.path[0][0]
			# self.connectivity = self.path[1][1]
			# self.bursters = self.path[2][1]
			# self.concentration = self.path[3][1].split('.csv')[0]
		self.data = np.genfromtxt(self.files[self.file_ind],delimiter=',',skip_header=1,skip_footer=1)
		with open(self.files[self.file_ind],'r') as file:
			pulled_data = file.read().split('\n')[0]
		self.headers = [pulled_data.split(',')[index] for index in range(len(pulled_data.split(','))) if index >= 1]
		self.path = self.files[self.file_ind].split(self.base_location)[1].split('/')
		for folder in range(len(self.path)):
			self.path[folder] = self.path[folder].split('_')
		self.voi = self.path[0][1]
		self.var = self.path[0][0]
		self.connectivity = self.path[1][1]
		self.bursters = self.path[2][1]
		self.concentration = self.path[3][1].split('.csv')[0]
		self.seed = self.headers[self.cell_ind].split(' ')[1]
		self.xdata = self.data[:,0]
		self.ydata = self.data[:,1+self.cell_ind]
		self.coordinates = np.column_stack((self.xdata,self.ydata))
		self.smoothed_coordinates = self.smooth_data()
		self.avg_dt = np.mean(self.xdata[1:]-self.xdata[0:-1])
		self.min_t = np.min(self.xdata)
		self.max_t = np.max(self.xdata)
		if reset_output == True:
			self.peaks = None
			self.periods = None
			self.avg_period = None
			self.frequency = None
			self.std = None
			self.stdm = None
			self.periods_calculated = False
			self.best_fit_polynomial = None
	
	def reset_data(self):
		self.xdata = self.data[:,0]
		self.ydata = self.data[:,1 + self.cell_ind]
		self.coordinates = np.column_stack((self.xdata,self.ydata))
		self.smoothed_coordinates = self.smooth_data()
		self.avg_dt = np.mean(self.xdata[1:]-self.xdata[0:-1])
		self.min_t = np.min(self.xdata)
		self.max_t = np.max(self.xdata)
		self.avg_period = None
		self.frequency = None
		self.std = None
		self.stdm = None
		self.periods_calculated = False
		self.best_fit_polynomial = None
	
	def replot_data(self,fit_line=False,reset_fit_line=False,reset_scatter=False,threshold=False,thresh_y=None,thresh_x=None,scatter=False,scatter_x=None,scatter_y=None):
		if self.smooth == False and self.display_both == False:
			self.graph.set_xdata(self.coordinates[:,0])
			self.graph.set_ydata(self.coordinates[:,1])
			self.smoothed_graph.set_visible(False)
			self.graph.set_visible(True)
		elif self.smooth == True and self.display_both == False:
			self.smoothed_graph.set_xdata(self.smoothed_coordinates[:,0])
			self.smoothed_graph.set_ydata(self.smoothed_coordinates[:,1])
			self.smoothed_graph.set_visible(True)
			self.graph.set_visible(False)
		elif self.display_both == True:
			self.graph.set_xdata(self.coordinates[:,0])
			self.graph.set_ydata(self.coordinates[:,1])
			self.smoothed_graph.set_xdata(self.smoothed_coordinates[:,0])
			self.smoothed_graph.set_ydata(self.smoothed_coordinates[:,1])
			self.smoothed_graph.set_visible(True)
			self.graph.set_visible(True)
		
		if fit_line == True:
			x_min = np.min(self.path[0])
			x_max = np.max(self.path[0])
			number_samples = (x_max - x_min)/self.avg_dt
			fit_line_xs = np.linspace(x_min,x_max,num=number_samples)
		
		if fit_line == True and self.best_fit_line == None:
			#create the best_fit_line
			self.best_fit_line, = self.ax9.plot(fit_line_xs,self.best_fit_polynomial(fit_line_xs),color = 'black')
		elif fit_line == True and self.best_fit_line != None:
			self.best_fit_line.set_xdata(fit_line_xs)
			self.best_fit_line.set_ydata(self.best_fit_polynomial(fit_line_xs))
		elif reset_fit_line == True:
			self.best_fit_line.set_xdata(0)
			self.best_fit_line.set_ydata(0)
		elif threshold == True:
			self.best_fit_line.set_xdata(thresh_x)
			self.best_fit_line.set_ydata(thresh_y)
		
		if scatter == True:
			self.plotted_peaks.remove()
			self.plotted_peaks = self.ax9.scatter(scatter_x,scatter_y,color='black')
		elif scatter == False and self.started == False:
			self.plotted_peaks.remove()
			self.plotted_peaks = self.ax9.scatter(np.nan,np.nan,color='black')
		
		if reset_scatter == True:
			self.plotted_peaks.remove()
			self.plotted_peaks = self.ax9.scatter(np.nan,np.nan,color='black')
		plt.draw()
	
	def horizontal_line_event_handling(self):
		self.cids = []
		cid = self.graph.figure.canvas.mpl_connect('button_press_event',self.horizontal_line)
		self.cids.append(cid)
	
	def custom_line_event_handling(self):
		self.cids = []
		cid1 = self.graph.figure.canvas.mpl_connect('button_press_event', self.custom_line_on_click)
		cid2 = self.graph.figure.canvas.mpl_connect('button_release_event', self.custom_line_on_release)
		cid3 = self.graph.figure.canvas.mpl_connect('motion_notify_event', self.custom_line_on_motion)
		self.cids.append(cid1)
		self.cids.append(cid2)
		self.cids.append(cid3)
	
	def remove_noise_event_handling(self):
		self.lasso = LassoSelector(self.ax9, self.lasso_on_select)
		self.cids = []
		cid = self.graph.figure.canvas.mpl_connect('key_press_event', self.remove_noise_key_press)
		self.cids.append(cid)
	
	def manual_event_handling(self):
		self.peaks = []
		self.cids = []
		cid = self.graph.figure.canvas.mpl_connect('button_press_event', self.manual_selection_on_click)
		cid2 = self.graph.figure.canvas.mpl_connect('key_press_event', self.manual_selection_on_key_press)
		self.cids.append(cid)
		self.cids.append(cid2)

	def automatic(self):
		if self.started == False:
			self.started = True
			type_coordinates = self.get_type_coordinates()
			self.initiate_automatic(type_coordinates)
		self.cids = []
		cid = self.graph.figure.canvas.mpl_connect('button_press_event',self.change_threshold_percent)
		cid2 = self.graph.figure.canvas.mpl_connect('key_press_event',self.change_threshold_percent_key_press)
		self.cids.append(cid)
		self.cids.append(cid2)

	def horizontal_line(self, event):
		if event.inaxes != self.ax9: return
		self.path = [[self.graph_minx,self.graph_maxx],[event.ydata,event.ydata]]
		self.best_fit_polynomial = interpolate.interp1d(self.path[0],self.path[1],kind='linear',bounds_error=False,fill_value='extrapolate')
		self.replot_data(fit_line=True)
		self.threshold_analysis()
	
	def custom_line_on_click(self, event):
		if event.inaxes != self.ax9:return
		if event.button == 1:
			self.press = True
			self.path = [[],[]]
			self.path[0].append(event.xdata)
			self.path[1].append(event.ydata)
		else:
			return
	
	def custom_line_on_motion(self, event):
		if self.press == None: return
		if event.inaxes != self.ax9: return
		previous_x = self.path[0][len(self.path[0])-1]
		self.path[0].append(event.xdata)
		self.path[1].append(event.ydata)
		self.best_fit_polynomial = interpolate.interp1d(self.path[0],self.path[1],kind='linear',bounds_error=False,fill_value='extrapolate')
		self.replot_data(fit_line=True)
	
	def custom_line_on_release(self, event):
		if event.button == 1:
			if self.press == None:
				return
			elif self.press == True:
				self.press = None
				self.best_fit_polynomial = interpolate.interp1d(self.path[0],self.path[1],kind='linear',bounds_error=False,fill_value='extrapolate')
				self.replot_data(fit_line=True)
				self.threshold_analysis()
			self.disconnect_connections()
		else:
			return
	
	def remove_noise_key_press(self, event):
		if event.key == 'enter':
			if self.smooth == True and self.display_both == False:
				self.find_peaks(self.smoothed_coordinates)
			elif self.smooth == False and self.display_both == False:
				self.find_peaks(self.coordinates)
			elif self.display_both == True:
				self.find_peaks(self.coordinates)
			self.peaks = np.array(self.peaks)
			self.replot_data()
			self.disconnect_connections()
			self.peak_analysis()

	def find_peaks(self, type_coordinates):
		slice1 = type_coordinates[0:-2,0]
		slice2 = type_coordinates[1:-1,0]
		slice3 = type_coordinates[2:,0]
		spike_starts = np.where(np.isnan(slice1) & np.isfinite(slice2) & np.isfinite(slice3))[0] + 1
		spike_ends = np.where(np.isfinite(slice1) & np.isfinite(slice2) & np.isnan(slice3))[0] + 1
		peak_indexes = np.where(np.isnan(slice1) & np.isfinite(slice2) & np.isnan(slice3))[0] + 1
		#determine the type of system that is present.
		self.peaks = None
		peaks_started = False
		if len(spike_starts) != len(spike_ends):
			if math.fabs(len(spike_ends)-len(spike_starts)) <= 1:
				#there are different numbers of spike starts and spike ends.
				if spike_ends[0] < spike_starts[0]: 
					#a spike was cut up in the beginning of the graph
					for index in range(len(spike_ends)):
						if index == 0:
							local_max = self.find_max(0,spike_ends[0])
						elif index != 0:
							local_max = self.find_max(spike_starts[index-1], spike_ends[index])
						if peaks_started == False:
							self.peaks = local_max
							peaks_started = True
						elif peaks_started == True:
							self.peaks = np.row_stack((self.peaks,local_max))
				elif spike_starts[len(spike_starts)-1] > spike_ends[len(spike_ends)-1]:
					#a spike was cut up at the end of the graph
					for index in range(len(spike_starts)):
						if index != len(spike_starts)-1:
							local_max = self.find_max(spike_starts[index], spike_ends[index])
						elif index == len(spike_starts)-1:
							local_max = self.find_max(spike_starts[index],type_coordinates.shape[0]-1)
						if peaks_started == False:
							self.peaks = local_max
							peaks_started = True
						elif peaks_started == True:
							self.peaks = np.row_stack((self.peaks,local_max))
				else:
					print('ERROR IN LENGTH OF THE SPIKE START AND SPIKE END LISTS')
			else:
				print('ERROR IN THE LENGTH OF THE SPIKE START AND SPIKE END LISTS')
				print('len(spike_starts) = ',len(spike_starts))
				print('len(spike_ends) = ',len(spike_ends))
		elif len(spike_starts) == len(spike_ends):
			#there are the same number of spike starts and spike ends.
			if spike_ends[0] < spike_starts[0]:
				# the system has spikes that are cut at the beginning and end of the graph
				for index in range(len(spike_ends)):
					if index == 0:
						local_max = self.find_max(0,spike_ends[0])
					elif index > 0 and index != len(spike_ends)-1:
						local_max = self.find_max(spike_starts[index-1], spike_ends[index])
					elif index > 0 and index == len(spike_ends)-1:
						local_max = self.find_max(spike_starts[index],type_coordinates.shape[0]-1)
					if peaks_started == False:
						self.peaks = local_max
						peaks_started = True
					elif peaks_started == True:
						self.peaks = np.row_stack((self.peaks,local_max))
			elif spike_starts[0] < spike_ends[0]:
				#the system is normal
				for index in range(len(spike_starts)):
					local_max = self.find_max(spike_starts[index], spike_ends[index])
					if peaks_started == False:
						self.peaks = local_max
						peaks_started = True
					elif peaks_started == True:
						self.peaks = np.row_stack((self.peaks,local_max))
			else:
				print('ERROR IN THE POSITIONING OF THE SPIKES IN THE SYSTEM')
	
	def manual_selection_on_click(self, event):
		if self.smooth == True:
			dx = self.smoothed_coordinates[:,0] - event.xdata
			dy = self.smoothed_coordinates[:,1] - event.ydata
			dx2 = dx * dx
			dy2 = dy * dy
			distances = np.sqrt(dx2 + dy2)
			peak = self.smoothed_coordinates[distances[:] == np.min(distances[:])][0]
			self.peaks.append(peak)
		elif self.smooth == False:
			dx = self.coordinates[:,0] - event.xdata
			dy = self.coordinates[:,1] - event.ydata
			dx2 = dx * dx
			dy2 = dy * dy
			distances = np.sqrt(dx2 + dy2)
			peak = self.coordinates[distances[:] == np.min(distances[:])][0]
			self.peaks.append(peak)
	
	def manual_selection_on_key_press(self, event):
		if event.key == 'enter':
			self.peaks = np.array(self.peaks)
			self.replot_data()
			self.peak_analysis()
			self.disconnect_connections()

	def change_threshold_percent(self, event):
		if event.inaxes != self.ax9: return
		if event.button == 1:
			y = (event.ydata - self.baseline_function(event.xdata)) / (self.max_y - self.min_y)
			if y >= self.min_y and y <= self.max_y:
				self.threshold_percent = y
				if self.smooth == True:
					type_coordinates = self.smoothed_coordinates
				elif self.smooth == False:
					type_coordinates = self.coordinates
				self.automatic_analysis(type_coordinates)

	def change_threshold_percent_key_press(self, event):
		if (event.key == 'up') or (event.key == 'down') or (event.key == 'alt+up') or (event.key == 'alt+down'):
			if event.key == 'up':
				self.threshold_percent += .01
			elif event.key == 'down':
				self.threshold_percent -= .01
			elif event.key == 'alt+up':
				self.threshold_percent += .1
			elif event.key == 'alt+down':
				self.threshold_percent -= .1
			if self.smooth == True:
				type_coordinates = self.smoothed_coordinates
			elif self.smooth == False:
				type_coordinates = self.coordinates
			self.automatic_analysis(type_coordinates)
		else:
			return
	
	def find_max(self, index1, index2):
		region = self.coordinates[index1:index2+1,1]
		local_max = self.coordinates[index1:index2+1,:][region == np.max(region)]
		return local_max
	
	def lasso_on_select(self, verts):
		path = Path(verts)
		if self.smooth == True and self.display_both == False:
			mask = np.array([path.contains_point(xy) for xy in self.smoothed_coordinates])
			pos = np.where(mask)[0]
			self.smoothed_coordinates[pos,0] = np.nan
			self.smoothed_coordinates[pos,1] = np.nan
			#self.coordinates = self.coordinates[mask,:]
			self.smoothed_graph.set_xdata(self.smoothed_coordinates[:,0])
			self.smoothed_graph.set_ydata(self.smoothed_coordinates[:,1])
		elif self.smooth == False and self.display_both == False:
			mask = np.array([path.contains_point(xy) for xy in self.coordinates])
			pos = np.where(mask)[0]
			self.coordinates[pos,0] = np.nan
			self.coordinates[pos,1] = np.nan
			#self.coordinates = self.coordinates[mask,:]
			self.graph.set_xdata(self.coordinates[:,0])
			self.graph.set_ydata(self.coordinates[:,1])
		elif self.display_both == True:
			mask = np.array([path.contains_point(xy) for xy in self.coordinates])
			pos = np.where(mask)[0]
			self.coordinates[pos,0] = np.nan
			self.coordinates[pos,1] = np.nan
			#self.coordinates = self.coordinates[mask,:]
			self.graph.set_xdata(self.coordinates[:,0])
			self.graph.set_ydata(self.coordinates[:,1])
			mask = np.array([path.contains_point(xy) for xy in self.smoothed_coordinates])
			pos = np.where(mask)[0]
			self.smoothed_coordinates[pos,0] = np.nan
			self.smoothed_coordinates[pos,1] = np.nan
			#self.coordinates = self.coordinates[mask,:]
			self.smoothed_graph.set_xdata(self.smoothed_coordinates[:,0])
			self.smoothed_graph.set_ydata(self.smoothed_coordinates[:,1])
		plt.draw()
	
	def threshold_analysis(self):
		zero_points = self.best_fit_polynomial(self.coordinates[:,0])
		current_points = self.coordinates[1:,:]
		last_points = self.coordinates[0:-1,:]
		slope = (current_points[0:,1] - last_points[0:,1]) / (current_points[0:,0] - last_points[0:,0])
		smoothed_slope,cross = self.get_slope(current_points[:,0], last_points[:,0],zero_points[1:])
		#self.peaks = self.coordinates[1:][(current_points[0:,1] >= zero_points[1:]) & (last_points[0:,1] <= zero_points[1:]) & (smoothed_slope >= 0) & (slope >= 0)]
		self.peaks = self.coordinates[1:][(smoothed_slope >= 0) & (cross == True)]
		self.peak_analysis()
	
	def get_derivative(self):
		data = self.smooth_data()
		self.derivative = interpolate.interp1d(data[:,0],data[:,1],kind='linear',bounds_error=False,fill_value='extrapolate')
	
	def get_slope(self, tf, ti, zero):
		self.get_derivative()
		yi = self.derivative(ti)
		yf = self.derivative(tf)
		dy = yf - yi
		dx = tf - ti
		slope = np.divide(dy,dx)
		cross = (yi <= zero) & (yf >= zero)
		return (slope,cross)
	
	def smooth_data(self):
		slice1 = self.coordinates[0:-4,:]
		slice2 = self.coordinates[1:-3,:]
		slice3 = self.coordinates[2:-2,:]
		slice4 = self.coordinates[3:-1,:]
		slice5 = self.coordinates[4:,:]
		new_data = slice1 + slice2 + slice3 + slice4 + slice5
		new_data = new_data * (1/5)
		return new_data
	
	def peak_analysis(self):
		if self.peaks.shape[0] > 1:
			self.periods = self.peaks[1:,0] - self.peaks[0:-1,0]
			peaks = self.peaks[1:][self.peaks[1:,0] - self.peaks[0:-1,0] > self.threshold]
			peaks = np.concatenate((peaks,self.peaks[0:-1][self.peaks[1:,0] - self.peaks[0:-1,0] > self.threshold]))
			self.periods = self.periods[self.periods > self.threshold]
			if self.periods.shape[0] > 1:
				self.periods_calculated = True
				self.avg_period = np.mean(self.periods)
				self.frequency = 1/self.avg_period
				self.std = np.std(self.periods)
				self.stdm = self.std / math.sqrt(len(self.periods))
				self.replot_data(scatter=True,scatter_x=peaks[:,0],scatter_y=peaks[:,1])
				self.update_output_label()
				self.disconnect_connections()
				self.mode()
				self.add_terminal_line('analysis complete')
			elif self.periods.shape[0] == 1 and self.peaks.shape[0] == 2:
				self.periods_calculated = True
				self.avg_period = np.mean(self.periods)
				self.frequency = 1/self.avg_period
				self.std = np.std(self.periods)
				self.stdm = self.std / math.sqrt(len(self.periods))
				self.replot_data(scatter = True, scatter_x = peaks[:,0], scatter_y = peaks[:,1])
				self.update_output_label()
				self.disconnect_connections()
				self.mode()
				self.add_terminal_line('analysis complete')
			else:
				self.replot_data(reset_scatter= True)
	
	def disconnect_connections(self):
		if len(self.cids) >= 1 and self.cids != []:
			for index in range(len(self.cids)):
				self.graph.figure.canvas.mpl_disconnect(self.cids[index])
			self.cids = []
		if self.mode == self.analysis_functions['remove noise']:
			self.lasso.disconnect_events()
	
	def continueous_key_bindings(self, event):
		if event.key in self.keyboard_shortcuts:
			key_func = self.keyboard_shortcuts[event.key]
			key_func(event)
			self.replot_data()
	
	def get_files(self):
		with open(self.files_list, 'r') as file:
			pulled_data = file.read().split('\n')
			for index in range(len(pulled_data)):
				if index >= 1:
					self.files.append(pulled_data[index])
				elif index == 0:
					self.base_location = '{}/'.format(pulled_data[index])
		self.output_file_name = 'output.csv'
		self.period_output_file_name = 'period_output.csv'

	def undo_append(self, event):
		status = False
		if self.log_ind != 0:
			status = True
			self.log_ind -= 1
		self.output_dict = self.output_dict_log[self.log_ind]
		if status == True:
			self.add_terminal_line('undo append')

	def redo_append(self, event):
		status = False
		if self.log_ind != len(self.output_dict_log) - 1:
			status = True
			self.log_ind += 1
		self.output_dict = self.output_dict_log[self.log_ind]
		if status == True:
			self.add_terminal_line('redo append')

	def append_periods(self, event):
		if self.var not in self.output_dict:
			self.output_dict[self.var] = OrderedDict()
		if self.voi not in self.output_dict[self.var]:
			self.output_dict[self.var][self.voi] = OrderedDict()
		if self.connectivity not in self.output_dict[self.var][self.voi]:
			self.output_dict[self.var][self.voi][self.connectivity] = OrderedDict()
		if self.bursters not in self.output_dict[self.var][self.voi][self.connectivity]:
			self.output_dict[self.var][self.voi][self.connectivity][self.bursters] = OrderedDict()
		if self.concentration not in self.output_dict[self.var][self.voi][self.connectivity][self.bursters]:
			self.output_dict[self.var][self.voi][self.connectivity][self.bursters][self.concentration] = self.periods
		elif self.concentration in self.output_dict[self.var][self.voi][self.connectivity][self.bursters]:
			period_list = self.output_dict[self.var][self.voi][self.connectivity][self.bursters][self.concentration]
			period_list = np.concatenate((period_list, self.periods))
			self.output_dict[self.var][self.voi][self.connectivity][self.bursters][self.concentration] = period_list
		if self.log_ind != len(self.output_dict_log)-1:
			dicts_to_remove = []
			for index in range(self.log_ind+1,len(self.output_dict_log)):
				dicts_to_remove.append(self.output_dict_log[index])
			for output_dictionary in dicts_to_remove:
				self.output_dict_log.remove(output_dictionary)
		self.output_dict_log.append(self.output_dict)
		if len(self.output_dict_log) > self.output_log_length:
			self.output_dict_log.remove(self.output_dict_log[0])
		self.log_ind = len(self.output_dict_log) - 1
		self.add_terminal_line('output appended')

	def write_periods(self, event):
		with open(self.period_output_file_name,'w') as file:
			lines = []
			for var in self.output_dict:
				for voi in self.output_dict[var]:
					for connectivity in self.output_dict[var][voi]:
						for bursters in self.output_dict[var][voi][connectivity]:
							for concentration in self.output_dict[var][voi][connectivity][bursters]:
								periods = self.output_dict[var][voi][connectivity][bursters][concentration]
								line = '{},{},{},{},{}'.format(var,voi,connectivity,bursters,concentration)
								for period in range(len(periods)):
									line = line +  ',' + str(periods[period])
								lines.append(line)
			for index in range(len(lines)):
				if index != len(lines)-1:
					lines[index] = lines[index] + '\n'
				file.write(lines[index])
		self.add_terminal_line('periods recorded')

	def write_results(self, event):
		with open(self.output_file_name,'w') as file:
			lines = []
			for var in self.output_dict:
				for voi in self.output_dict[var]:
					for connectivity in self.output_dict[var][voi]:
						for bursters in self.output_dict[var][voi][connectivity]:
							for concentration in self.output_dict[var][voi][connectivity][bursters]:
								periods = self.output_dict[var][voi][connectivity][bursters][concentration]
								avg_period = np.mean(periods)
								frequency = 1/avg_period
								std = np.std(periods)
								stdm = std / math.sqrt(len(periods))
								line = '{},{},{},{},{},{},{},{},{}'.format(var,voi,connectivity,bursters,concentration,avg_period,frequency,std,stdm)
								lines.append(line)
			for index in range(len(lines)):
				if index != len(lines)-1:
					lines[index] = lines[index] + '\n'
				file.write(lines[index])
		self.add_terminal_line('results recorded')

	def toggle_smooth(self, event):
		if self.smooth == True:
			self.smooth = False
			line = 'reduced noise: off'
		elif self.smooth == False:
			self.smooth = True
			line = 'reduced noise: on'
		self.check_automatic()
		self.add_terminal_line(line)

	def check_automatic(self):
		if self.started == True:
			type_coordinates = self.get_type_coordinates()
			self.initiate_automatic(type_coordinates)

	def toggle_both(self, event):
		if self.display_both == True:
			self.display_both = False
			line = 'display both: off'
		elif self.display_both == False:
			self.display_both = True
			line = 'display both: on'
		self.add_terminal_line(line)

	def automatic_analysis(self, type_coordinates):
		fixed_coordinates = np.copy(self.fixed_coordinates)
		threshold = self.min_y + ((self.max_y - self.min_y)*self.threshold_percent)
		fixed_coordinates[:,1][self.fixed_coordinates[:,1] <= threshold] = 0
		thresh_c = np.copy(fixed_coordinates)
		thresh_c[:,1] = threshold + self.baseline
		indexes = peakutils.indexes(fixed_coordinates[:,1], thres=0, min_dist=self.threshold/self.avg_dt)
		self.peaks = None
		self.peaks = type_coordinates[indexes,:]
		self.replot_data(threshold=True,thresh_x=thresh_c[:,0],thresh_y=thresh_c[:,1])
		self.check_for_sigh(fixed_coordinates)
		self.peak_analysis()

	def change_baseline_order(self, event):
		self.baseline_order = self.baseline_order + event.step
		if self.baseline_order < self.min_baseline_order:
			self.baseline_order = self.min_baseline_order
		self.add_terminal_line('baseline order = {}'.format(self.baseline_order))
		type_coordinates = self.get_type_coordinates()
		self.initiate_automatic(type_coordinates)

	def increase_baseline_order(self, event):
		self.baseline_order = self.baseline_order + 1
		if self.baseline_order < self.min_baseline_order:
			self.baseline_order = self.min_baseline_order
		elif self.baseline_order > self.max_baseline_order:
			self.baseline_order = self.max_baseline_order
		self.add_terminal_line('baseline order = {}'.format(self.baseline_order))
		type_coordinates = self.get_type_coordinates()
		self.initiate_automatic(type_coordinates) 

	def decrease_baseline_order(self, event):
		self.baseline_order = self.baseline_order - 1
		if self.baseline_order < 0:
			self.baseline_order = 0
		elif self.baseline_order > self.max_baseline_order:
			self.baseline_order = self.max_baseline_order
		self.add_terminal_line('baseline order = {}'.format(self.baseline_order))
		type_coordinates = self.get_type_coordinates()
		self.initiate_automatic(type_coordinates) 

	def add_terminal_line(self, line):
		self.terminal_lines.append(line)
		if len(self.terminal_lines) == self.number_terminal_lines + 1:
			self.terminal_lines.remove(self.terminal_lines[0])
		self.write_terminal()

	def initiate_automatic(self, type_coordinates):
		self.started = True
		self.baseline = peakutils.baseline(type_coordinates[:,1],self.baseline_order)
		baseline_c = np.copy(type_coordinates)
		baseline_c[:,1] = self.baseline
		self.baseline_function = interpolate.interp1d(baseline_c[:,0],baseline_c[:,1],kind='linear',bounds_error=False,fill_value='extrapolate')
		self.fixed_coordinates = np.copy(type_coordinates)
		self.fixed_coordinates[:,1] = self.fixed_coordinates[:,1] - self.baseline
		self.min_y = np.min(self.fixed_coordinates[:,1])
		self.max_y = np.max(self.fixed_coordinates[:,1])
		self.automatic_analysis(type_coordinates)

	def get_type_coordinates(self):
		if self.smooth == True:
			return self.smoothed_coordinates
		elif self.smooth == False:
			return self.coordinates

	def load_periods(self, event):
		with open(self.period_output_file_name,'r') as file:
			pulled_data = file.read().split('\n')
		for index in range(len(pulled_data)):
			pulled_data[index] = pulled_data[index].split(',')
			var = pulled_data[index][0]
			voi = pulled_data[index][1]
			connectivity = pulled_data[index][2]
			bursters = pulled_data[index][3]
			concentration = pulled_data[index][4]
			periods = np.array([float(pulled_data[index][x]) for x in range(5,len(pulled_data[index]))])
			if var not in self.output_dict:
				self.output_dict[var] = OrderedDict()
			if voi not in self.output_dict[var]:
				self.output_dict[var][voi] = OrderedDict()
			if connectivity not in self.output_dict[var][voi]:
				self.output_dict[var][voi][connectivity] = OrderedDict()
			if bursters not in self.output_dict[var][voi][connectivity]:
				self.output_dict[var][voi][connectivity][bursters] = OrderedDict()
			if concentration not in self.output_dict[var][voi][connectivity][bursters]:
				self.output_dict[var][voi][connectivity][bursters][concentration] = periods
			elif concentration in self.output_dict[var][voi][connectivity][bursters]:
				period_list = self.output_dict[var][voi][connectivity][bursters][concentration]
				period_list = np.concatenate((period_list,periods))
		self.add_terminal_line('periods loaded')



def get_params():
	with open('frequency calculator parameters.txt','r') as file:
		pulled_data = file.read().split('\n')
	params = [pulled_data[index].split('=')[1].strip() for index in range(len(pulled_data)) if len(pulled_data[index].split('=')) > 1]
	return params

if __name__ == '__main__':
	calculators = {'experimental':experimental_calculator,'computational':computational_calculator}
	parameters = get_params()
	calculator = calculators[parameters[0]]
	parameters = parameters[1:]
	app = calculator(parameters)
																	
				
