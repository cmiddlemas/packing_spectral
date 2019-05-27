"""
Script to make a scatter plot of data stored in tab separated columns
in a text file.

Author: Timothy Middlemas (tm17@princeton.edu)

Dependencies:
    - Python3 interpreter
    - Matplotlib

Usage: scatter_plot.py <infile> [x_axis] [y_axis]

where

infile = file containing data, optional, read from stdin if not specified
x_axis = string to label x-axis
y_axis = string to label y-axis

Output: A pyplot object sent to window manager with the scatter plot

example file format 
(# is a comment, do not include, there is no support for comments in actual parser)
((tab) tells you put a tab whitespace, ignore actual spaces in following)
(similarly (newline) tells you to put a newline):

0.05 (tab) 0.9 (newline) # x, y
... # Continue with more data points


Options:
    --help: Print this message
"""

import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Read command line
    for entry in sys.argv:
        if entry == "--help":
            print(__doc__)
            sys.exit()
    if len(sys.argv) == 4:
        infile = open(sys.argv[1])
        x_axis = sys.argv[2]
        y_axis = sys.argv[3]
    elif len(sys.argv) == 3:
        infile = sys.stdin
        x_axis = sys.argv[1]
        y_axis = sys.argv[2]
    else:
        raise Exception("You must provide 2 or 3 arguments.")

    # Parse input file/stream into data lists
    data_x = []
    data_y = []
    for i, line in enumerate(infile):
        list_of_words = line.rsplit()
        data_x.append(float(list_of_words[0]))
        data_y.append(float(list_of_words[1]))

    infile.close()
    
    # Plot the data
    plt.scatter(data_x, data_y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()
