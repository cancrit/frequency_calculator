This application is designed to take the files that are listed in file_list.txt
and plot the data using a graphical user interface that allows the user to peroform
complex analysis of the data.

====================================================================================
======================== analysis methods ==========================================
====================================================================================

This program offers 5 different analysis methods for the user to take advantage of.

1.) Automatic analysis: This method allows the user to take advantage of a method
that will automatically analyze the data. This automatic analysis method is
described in further detail in the detailed analysis description section. The
procedure for using this method is described in the analysis procedure section of
this document.

2.) Horizontal line analysis: This allows the user to specify a single threshold
value on the y axis that can be used to calculate the periods of the data on the
graph. Further details will be explained in the analysis procedure section of this
document.

3.) Custom line analysis: This allows the user to specify a custom function for
the threshold values that can be used to calculate the periods of the data on the
graph. Further details will be explained in the analysis procedure section of this
document.

4.) Noise removal analysis: This allows the user to remove any parts of the graph
that the user deem neccisary in order to extract the peaks from the noise. The
program can then use the remaining datapoints to calculate the periods of the data.

5.) Manual selection analysis: This allows the user to manualy select the locations
of the peaks in the data. The program then searches for the datapoints that 
correspond to the selected locations and uses those datapoints to calculate the
periods of the data.

====================================================================================
======================== keyboard shortcuts ========================================
====================================================================================

There are numerous keyboard shortcuts that can be used at all times while the
program is running. Some of the functions that have keyboard shortcuts associated
with them that can also be performed in other ways, and there are some that cannot.

shortcuts:
	'ctrl + a' = Add calculated periods to the list of periods that the program has
	             stored in memory.*

	'ctrl + q' = Record the list of periods that the program has stored in memory in
	             a csv file that the user can acess later.**

	'ctrl + shift + q' = Calculate and record the proper output values using the 
	                     list of periods that the program has stored in memory.**

	'ctrl + u' = Remove a set of periods that was added to the list of periods that 
	             the program has stored in memory.

	'ctrl + r' = Add a set of periods that was removed back to the list of periods 
	             that the program has stored in memory.

	'ctrl + shift + s' = Toggle reduced noise mode.***

	'ctrl + alt + s' = Toggle display both reduced noise and normal mode.

	'ctrl + right' = Switch to the next file in the file list.****

	'ctrl + left' = Switch to the previous file in the file list.****

	'ctrl + up' = Switch to the next column of the current file.****

	'ctrl + down' = Switch to the previous column of the current file.****

	'ctrl + l' = Load the periods that were last output by the program.*****

	'left' or 'mouse wheel down' = Decrease baseline order.******

	'right' or 'mouse wheel up' = Increase baseline order.******

	'up' = Move the analysis line up a small amount.******

	'down' = Move the analysis line down a small amount.******

* This shortcut does not record the list of periods in a file. Therefore unless one
  of the other keyboard shortcuts is used the list of periods will be lost when the
  window is closed.
** These shortcuts record the information in the same file and as a result will
   overwrite whatever is in the file everytime they are used. This means that these
   shortcuts can be used as often as the user would like without the user having to
   worry about sacrificing performance.
*** This method allows the user to toggle between the reduced noise graph and the
	original graph modes. However it is important to keep in mind that the
	horizontal line analysis and custom line analysis modes will produce the same
	results regardless of which mode you are in while the noise removal and manual
	selection analysis methods work differently depending on the which mode the
	program is in.
**** These shortcuts are set up on loops meaning if the next command is used and
     there is not another column or file for it to switch to it will loop back
     around to the beginning of the list.
***** This command is particularly useful for when the user has done large amounts
      of analysis and does not want to lose all of their progress.
****** These commands are only used with the automatic analysis mode. Changing the
       baseline allows the user to get a better fit for the data. For further
       information on how this works see the detailed analysis description section. 

====================================================================================
===================== user interface descriptions ==================================
====================================================================================

The user interface of this program provides the user with many useful tools for the
analysis of the data. In this section these different tools will be described.

At the top of the window there are four different text boxes where information
relating to the location in the file list and location within the data file is 
displayed. From the left the first text box displays a partial path to the file that
the data graphed is from. The second text box displays the current file number and
the total number of files. The third text box displays the value that is in row 1 of
the current column of the current csv file. The fourth text box displays the current
column number and the total number of columns in the csv file that contain data that
needs to be analyzed.

In the middle section of the window there are two regions of interest. On the right
side of the window there is the graph where most of the interaction with the data
will take place. On the left side of the window there are three boxes that contain
the tools for processing the data. The box at the top allows the user to select the
mode of analysis that they would like to use. The middle box displays the output
information for the given data. If no analysis has been done then these values will
all display None. This allows the user to check to see if the results of the
analysis seem reasonable before the periods from the current data set are added to
the list of periods that the program has stored in memory. The bottom box allows the
user to see information regarding what has been done in the background. This is
important because it allows the user to have confirmation that a button they clicked
on or a keyboard shortcut that they used was succesfully registered. It can also
help the user keep track of what has and has not been done to the data in case of
distractions.

The bottom of the screen provides the user with 9 buttons that can be used to both
manage the data that is to be output from the analysis and to navigate the file
list and the columns of data within the current file. The buttons used for managing
the output data all have keyboard shortcuts associated with them that were described
above in the keyboard shorcuts section of this document. (An important note for
users that do not have extensive knowledge of programming. The word append that is 
used on these 9 buttons means to add something to a list. So append periods means to
add the periods from the current data set to the master list of periods.)

====================================================================================
======================== analysis procedure ========================================
====================================================================================

This section provides some basic instruction on the general procedure for analyzing
the data.

The first step in the analysis process is for the user to select the appropriate
analysis method for the data that is currently being displayed. Each of the
analysis methods have their own unique set of controls.

1.) Automatic analysis: This feature will automatically produce a line that is based
on the value of the baseline order and determine the locations of peaks that are
used to automatically analyze the data. However, it is possible to move the line up
and down on the graph. This can be done in two different ways the first method is to
left click on the graph, but this method is a little bit difficult to control. For
this reason it is advisable that the user take advantage of the keyboard shortcuts
mentioned in the keyboard shortcuts section of this document.

2.) Horizontal line analysis: Left click on the graph at the desired hieght for the
threshold value needed to analyze the data. Then a horizontal line will be displayed
and the output values for the analysis with that threshold will be displayed on the
left side of the window in the appropriate text box.

3.) Custom line analysis: Left click on the graph where you would like the
custom threshold line to start. Then drag the mouse along the path of the custom
threshold line, and when you reach the end of the line release the left mouse button
as with the horizontal line analysis a threshold line will be displayed and the
output values will be displayed on the left sid of the window.

4.) Noise removal analysis: Click and drag to trace out what parts of the graph
should be removed. Upon releasing the mouse button the program will remove any data
points that are within that region from the graph. This can be done as many times
as the user sees fit. Once the noise has been removed press 'enter' or 'return'
depending on which type of opperating system you are using. The output data will
then be displayed on the left side of the window.

5.) Manual selection analysis: Click on a position as close as possible to the
desired peak in order to select that peak. Once this has been done for all of the
peaks press 'enter' or 'return' depending on which opperating system you are using.
Doing so will perform the analysis and display the output data on the left side of
the window.

Once the output data on the left side of the window is satifactory the user can
use the buttons at the bottom of the screen or the keyboard shortcuts may be used
to manage the ouput data from the current set of data.

If graph does not look like it is usable it is possible to skip the graph and move
on to the next set of data using the buttons in the bottom right portion of the
window.

====================================================================================
======================== file layout ===============================================
====================================================================================

The data that this program uses must come from the file_list.txt file. However there
is also one more condition that must be followed and that is that the file paths for
the files must follow the following layout.

where
[K] = The concentration of potassium in mM
drug = The name or shorthand for the drug that is being used so that you will know
       what it is
base = The path to the folder were all of the data is stored which should also be
       the first line of the file_list.txt file
folder = An arbitrary folder name

base/drug/folder/[K] mM.csv

* File paths are like directions to the file. Everything seperated by '/' is 
  considered to be a folder within the previously listed folder. Unless it has a
  file extension on it like ".csv" in which case it is a file within the previously
  listed folder.

It is important that this layout is followed because the output from the program
uses whatever is in the positions were drug and [K] are as the identity of the drug
and the concentration of potassium ions. Meaning that if this format is not followed
you will likely end up with useless information as output from the program.

As far as the layout of the csv goes there is an example csv file included that
demonstrates the layout requuired for the csv files. (Make sure that they are
actually ".csv" files and not one of excels other file formats as many of them will 
be unreadable by this program.)

It is also important that you have the peakutils folder in the same folder as this
program. There are also many additional things that can be custimized using the
frequency calculator paramaters file. This program requires the parameters file be
located in the same place as this program.

====================================================================================
======================== detailed analysis description =============================
====================================================================================

This section provides descriptions of the different features of the program that are
intended for advanced users or users with programming knowledge.

1.) Automatic analysis: This method uses the value of the baseline order to come up
with a polynomial expression that represents the shape of the drift in the baseline.
It then subtracts the drift of the baseline from the data which flattens the data.
The red line that is displayed on the screen is a representation of the polynomial
that the program is using to model the drift but with a position of the graph that
is specified by the user. Once the data has been flattened the program ignores any
data that is below the red line and uses the functions defined in the peakutils
package to determine the location of peaks in the data that is above the red line.

2.) Horizontal line analysis: This method uses a horizontal line with a position
given by the user to determine where the data after smoothing crosses the threshold
with a positive slope and treats this as a peak.

3.) Custom line analysis: This method is identical to the method used in the
horizontal line analysis except in this method the line is not a horizontal line.
Rather it is a line that was drawn by the user of the program.

4.) Remove noise analysis: This method takes an area drawn by the user and removes
all of the data that is present within the area. Once all noise has been removed
and the user is satisfied the program uses a method of peak detection similar to
the one used to the automatic analysis to determine the peaks present in any of the
data that remains.

5.) Mannual selection analysis: This method searches the data set for the closest
point to the location specified by the user and uses this as the location of a peak.

Any further questions that you may have can be answered by either looking at the
code for the program yourself or by contacting the developer using the contact
information provided in the contact information section of this document.

====================================================================================
======================== contact information =======================================
====================================================================================

This application was developed and tested by Jacob Barfield. If you have any
questions, comments, or concerns you can reach me at the following email address
jhbarfield@mail.roanoke.edu
