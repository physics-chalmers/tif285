#!/usr/bin/env python
# coding: utf-8

# # Python and Jupyter notebooks: part 01
# 
# Original version: Dick Furnstahl, Ohio State University<br/>
# Last revised: 29-Aug-2019 by Christian Forssén [christian.forssen@chalmers.se]

# **You can find valuable documentation under the Jupyter notebook Help menu. The "User Interface Tour" and "Keyboard Shortcuts" are useful places to start, but there are also many other links to documentation there.** 

# This is a whirlwind tour of just the minimum we need to know about Python and Jupyter notebooks to get started doing data analysis.  We'll add more features and details as we proceed.
# 
# A Jupyter notebook is displayed on a web browser on a computer, tablet (e.g., IPad), or even your smartphone.  The notebook is divided into *cells*, of which two types are relevant for us:
# * Markdown cells: These have headings, text, and mathematical formulas in $\LaTeX$ using a simple form of HTML called markdown.
# * Code cells: These have Python code (or other languages, but we'll stick to Python).
# 
# Either type of cell can be selected with your cursor and will be highlighted in color when active.  You evaluate an active cell with shift-return (as with Mathematica) or by pressing `Run` on the toolbar.  Some notes:
# * When a new cell is inserted, by default it is a Code cell and will have `In []:` in front.  You can type Python expressions or entire programs in a cell.  How you break up code between cells is your choice and you can always put Markdown cells in between.  When you evaluate a cell it gets the next number, e.g., `In [5]:`.
# * On the menu bar is a pulldown menu that lets you change back and forth between Code and Markdown cells.  Once you evaluate a Markdown cell, it gets formatted (and has a blue border).  To edit the Markdown cell, double click in it. 
# 
# **Try double-clicking on this cell and then shift-return.**  You will see that a bullet list is created just with an asterisk and a space at the beginning of lines (without the space you get *italics* and with two asterisks you get **bold**).  **Double click on the title header above and you'll see it starts with a single #.**  Headings of subsections are made using ## or ###.  See this [Markdown cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) for a quick tour of the Markdown language (including how to add links!).
# 
# **Now try turning the next (empty) cell to a Markdown cell and type:** `Einstein says $E=mc^2$` **and then evaluate it.**  This is $\LaTeX$! (If you forget to convert to Markdown and get `SyntaxError: invalid syntax`, just select the cell and convert to Markdown with the menu.)

# In[ ]:





# The menus enable you to rename your notebook file (always ending in `.ipynb`) or `Save and Checkpoint` to save the changes to your notebook.  You can insert and delete cells (use the up and down arrows in the toolbar to easily move cells).  You will often use the `Kernel` menu to `Restart` the notebook (and possibly clear output). The buttons below the menu allow quick access to several useful commands. 

# As you get more proficient working with notebooks, you will certainly start using the shortcut keys from the command mode of cells. A cell that is marked in blue implies that you are in command mode. You can start editing the cell by hitting `Enter` (or by clicking inside it). You can exit from edit mode into command mode by hitting `Esc`. A list of shortcut keys can be seen when opening the command palette by clicking on the keyboard button.

# ## Ok, time to try out Python expressions and numpy
# 
# We can use the Jupyter notebook as a super calculator much like Mathematica and Matlab.  **Try some basic operations, modifying and evaluating the following cells, noting that exponentiation is with** `**` **and not** `^`.

# In[1]:


1 + 1  # Everything after a number sign / pound sign / hashtag) 
       #  is a comment


# In[2]:


3.2 * 4.713


# Note that if we want a floating point number (which will be the same as a `double` in C++), we *always* include a decimal point (even when we don't have to) while a number without a decimal point is an integer.

# In[3]:


3.**2


# We can define integer, floating point, and string variables, perform operations on them, and print them.  Note that we don't have to predefine the type of a variable and we can use underscores in the names (unlike Mathematica).  **Evaluate the following cells and then try your own versions.** 

# In[4]:


x = 5.
print(x)
x   # If the last line of a cell returns a value, it is printed.


# In[5]:


y = 3.*x**2 - 2.*x + 7.
print('y = ', y)           # Strings delimited by ' 's


# There are several ways to print strings that includes variables from your code. We recommend using the relatively newly added `fstring`. See, e.g., this [blog](https://cito.github.io/blog/f-strings/) for examples. 

# In[6]:


print(f'y = {y:.0f}')      # Just a preview: more on format later 
print(f'y = {y:.2f}')      #  (note that this uses the "new" fstring)


# The `fstring` will be used predominantly in this course, but you might also encounter older formatting syntax.

# In[7]:


print('x = %.2f  y = %.2f' %(x,y)) 
print('x = {0:.2f}  y = {1:.2f}'.format(x, y)) 
print(f'x = {x:.2f}  y = {y:.2f}')


# In[8]:


first_name = 'Christian'     # Strings delimited by ' 's
last_name = 'Forssén'
full_name = first_name + ' ' + last_name  # you can concatenate strings 
print(full_name)
# or
print(f'{first_name} {last_name}')


# Ok, how about square roots and trigonometric functions and ... 
# 
# *(Note: the next cells will give error messages --- keep reading to see how to fix them.)*

# In[9]:


sqrt(2)


# In[ ]:


sin(pi)


# We need to `import` these functions through the numpy library. There are other choices, but numpy works with the arrays we will use.  Note: *Never* use `from numpy import *` instead of `import numpy as np`.  Here `np` is just a abbreviation for numpy (which we can choose to be anything, but `np` is conventional).

# In[ ]:


import numpy as np


# In[ ]:


print(np.cos(0.))


# Now functions and constants like `np.sqrt` and `np.pi` will work.  Go back and fix the square root and sine.

# ### Debugging aside . . .
# 
# Suppose you try to import and it fails (**go ahead and evaluate the cell**):

# In[ ]:


import numpie


# When you get a `ModuleNotFoundError`, the first thing to check is whether you have misspelled the name. Try using Google, e.g., search for "python numpie". In this case (and in most others), Google will suggest the correct name (here it is numpy).  If the name does exist, check whether it sounds like the package you wanted.
# 
# If you have the correct spelling, check whether you have installed the relevant package.  If you installed Python with Anaconda (which we will assume you did -- if not, do it!), then use `conda list`, e.g., `conda list numpy` in a Terminal window (on a Mac or Linux box) or in an Anaconda Prompt window (on a Windows PC).

# ### numpy arrays
# 
# We will often use numpy arrays so we'll start with those.  They are *like* lists delimited by square brackets, i.e., `[]`s, and we will construct them with `np.arange(min, max, step)` to get an array from `min` to `max` in steps of `step`. Examples:

# In[ ]:


t_pts = np.arange(0., 10., .1)
t_pts


# If we give a numpy array to a function, each term in the list is evaluated with that function:

# In[ ]:


x = np.arange(1., 5., 1.)
print(x)
print(x**2)
print(np.sqrt(x))


# We can pick out elements of the list.  Why does the last one fail? 

# In[ ]:


print(x[0])
print(x[3])
print(x[4])


# ## Getting help
# 
# You will often need help identifying the appropriate Python (or numpy or scipy or ...) command or you will need an example of how to do something or you may get an error message you can't figure out.  In all of these cases, Google (or equivalent) is your friend. Always include "python" in the search string (or "numpy" or "matplotlib" or ...) to avoid getting results for a different language. You will usually get an online manual as one of the first responses if you ask about a function; these usually have examples if you scroll down. Otherwise, answers from *Stack Overflow* queries are your best bet to find a useful answer.

# ## Functions
# 
# There are many Python language features that we will use eventually, but in the short term what we need first are functions.  Here we first see the role of *indentation* in Python in place of {}s or ()s in other languages.  We'll always indent four spaces (never tabs!).  We know a function definition is complete when the indentation stops. 
# 
# To find out about a Python function or one you define, put your cursor on the function name and hit shift+Tab+Tab. **Go back and try it on `np.arange`.**  

# In[ ]:


# Use "def" to create new functions.  
#  Note the colon and indentation (4 spaces).
def my_function(x):
    """This function squares the input.  Always include a brief description
        at the top between three starting and three ending quotes.  We will
        talk more about proper documentation later.
        Try shift+Tab+Tab after you have evaluated this function.
    """
    return x**2

print(my_function(5.))

# We can pass an array to the function and it is evaluated term-by-term.
x_pts = np.arange(1.,10.,1.)
print(my_function(x_pts))


# In[ ]:


# Two variables, with a default for the second
def add(x, y=4.):
    """Add two numbers."""
    print("x is {} and y is {}".format(x, y))
    return x + y  # Return values with a return statement

# Calling functions with parameters
print('The sum is ', add(5, 6))  # => prints out "x is 5 and y is 6" and returns 11

# Another way to call functions is with keyword arguments
add(y=6, x=5)  # Keyword arguments can arrive in any order.


# How do you explain the following result?

# In[ ]:


add(2)


# ### Debugging aside . . .
# 
# There are two bugs in the following function.  **Note the line where an error is first reported and fix the bugs sequentially (so you see the different error messages).**

# In[ ]:


def hello_function()
    msg = "hello, world!"
    print(msg)
     return msg


# ## Plotting with Matplotlib
# 
# Matplotlib is the plotting library we'll use, at least at first.  We'll follow convention and abbreviate the module we need as `plt`.  The `%matplotlib inline` line tells the Jupyter notebook to make inline plots (we'll see other possibilities later).

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt


# Procedure we'll use to make the skeleton plot:
# 0. Generate some data to plot in the form of arrays.
# 1. Create a figure;
# 2. add one or more subplots;
# 3. make a plot and display it.

# In[ ]:


t_pts = np.arange(0., 10., .1)     # step 0.
x_pts = np.sin(t_pts)  # More often this would be from a function 
                       #  *we* write.

my_fig = plt.figure()              # step 1.
my_ax = my_fig.add_subplot(1,1,1)  # step 2: rows=1, cols=1, 1st subplot
my_ax.plot(t_pts, x_pts)           # step 3: plot x vs. t


# NOTE: When making just a single plot, you will more usually see steps 1 to 3 compressed into `plt.plot(t_pts, np.sin(t_pts))`.  Don't do this.  It saves a couple of lines but restricts your ability to easily extend the plot, which is what we want to make easy.

# We can always go back and dress up the plot:

# In[ ]:


my_fig = plt.figure()
my_ax = my_fig.add_subplot(1,1,1)  # nrows=1, ncols=1, first plot
my_ax.plot(t_pts, x_pts, color='blue', linestyle='--', label='sine')

my_ax.set_xlabel('t')
my_ax.set_ylabel(r'$\sin(t)$')  # here $s to get LaTeX and r to render it
my_ax.set_title('Sine wave')

# here we'll put the function in the call to plot!
my_ax.plot(t_pts, np.cos(t_pts), label='cosine')  # just label the plot

my_ax.legend();  # turn on legend


# Now make two subplots:

# In[ ]:


y_pts = np.exp(t_pts)         # another function for a separate plot

fig = plt.figure(figsize=(10,5))  # allow more room for two subplots

# call the first axis ax1
ax1 = fig.add_subplot(1,2,1)  # one row, two columns, first plot
ax1.plot(t_pts, x_pts, color='blue', linestyle='--', label='sine')
ax1.plot(t_pts, np.cos(t_pts), label='cosine')  # just label the plot
ax1.legend()

ax2 = fig.add_subplot(1,2,2)  # one row, two columns, second plot
ax2.plot(t_pts, np.exp(t_pts), label='exponential')  
ax2.legend();


# ### Saving a figure
# Saving a figure to disk is as simple as calling [`savefig`](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.savefig) with the name of the file (or a file object). The available image formats depend on the graphics backend you use.

# Let us save the figure (named 'fig') from the previous cell

# In[ ]:


fig.savefig("sine_and_exp.png")
# and a transparent version:
fig.savefig("sine_and_exp_transparent.png", transparent=True)


# ### Further examples with matplotlib
# 
# * The [matplotlib gallery](https://matplotlib.org/gallery.html) is a good resource for learning by working examples.
# * You are also welcome to explore the more extensive matplotlib tutorial by Geron. It is reproduced, with the accompanying open-source Apache License, in the [handson-ml-notebooks](handson-ml-notebooks/tools_matplotlib.ipynb) directory.

# ## Widgets!
# 
# A widget is an object such as a slider or a check box or a pulldown menu.  We can use them to make it easy to explore different parameter values in a problem we're solving, which is invaluable for building intuition.  They act on the argument of a function.  We'll look at a simple case here but plan to explore this more as we proceed.
# 
# The set of widgets we'll use here (there are others!) is from `ipywidgets`; we'll conventionally import the module as `import ipywidgets as widgets` and we'll also often use `display` from `Ipython.display`.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display

get_ipython().run_line_magic('matplotlib', 'inline')


# The simplest form is to use `interact`, which we pass a function name and the variables with ranges.  By default this makes a *slider*, which takes on integer or floating point values depending on whether you put decimal points in the range. **Try it! Then modify the function and try again.**

# In[ ]:


# We can do this to any function
def test_f(x=5.):
    """Test function that prints the passed value and its square.
       Note that there is no return value in this case."""
    print ('x = ', x, ' and  x^2 = ', x**2)
    
widgets.interact(test_f, x=(0.,10.));


# In[ ]:


# Explicit declaration of the widget (here FloatSlider) and details
def test_f(x=5.):
    """Test function that prints the passed value and its square.
       Note that there is no return value in this case."""
    print ('x = ', x, ' and  x^2 = ', x**2)
    
widgets.interact(test_f, 
                 x = widgets.FloatSlider(min=-10,max=30,step=1,value=10));


# Here's an example with some bells and whistles for a plot.  **Try making changes!**

# In[ ]:


def plot_it(freq=1., color='blue', lw=2, grid=True, xlabel='x', 
            function='sin'):
    """ Make a simple plot of a trig function but allow the plot style
        to be changed as well as the function and frequency."""
    t = np.linspace(-1., +1., 1000)  # linspace(min, max, total #)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)

    if function=='sin':
        ax.plot(t, np.sin(2*np.pi*freq*t), lw=lw, color=color)
    elif function=='cos':
        ax.plot(t, np.cos(2*np.pi*freq*t), lw=lw, color=color)
    elif function=='tan':
        ax.plot(t, np.tan(2*np.pi*freq*t), lw=lw, color=color)

    ax.grid(grid)
    ax.set_xlabel(xlabel)
    
widgets.interact(plot_it, 
                 freq=(0.1, 2.), color=['blue', 'red', 'green'], 
                 lw=(1, 10), xlabel=['x', 't', 'dog'],
                 function=['sin', 'cos', 'tan'])
    


# ## Numpy linear algebra
# 
# Having used numpy arrrays to describe vectors, we are now ready to try out matrices. We can define a $3 \times 3 $ real matrix $\hat{A}$ as

# In[1]:


import numpy as np
A = np.log(np.array([ [4.0, 7.0, 8.0], [3.0, 10.0, 11.0], [4.0, 5.0, 7.0] ]))
print(A)


# If we use the `shape` attribute we would get $(3, 3)$ as output, that is verifying that our matrix is a $3\times 3$ matrix. 

# In[5]:


A.shape


# We can slice the matrix and print for example the first column (Python organized matrix elements in a row-major order, see below) as

# In[ ]:


A = np.log(np.array([ [4.0, 7.0, 8.0], [3.0, 10.0, 11.0], [4.0, 5.0, 7.0] ]))
# print the first column, row-major order and elements start with 0
print(A[:,0])


# We can continue this was by printing out other columns or rows. The example here prints out the second column

# In[ ]:


A = np.log(np.array([ [4.0, 7.0, 8.0], [3.0, 10.0, 11.0], [4.0, 5.0, 7.0] ]))
# print the first column, row-major order and elements start with 0
print(A[1,:])


# Numpy contains many other functionalities that allow us to slice, subdivide etc etc arrays. We strongly recommend that you look up the [Numpy website for more details](http://www.numpy.org/). Useful functions when defining a matrix are the `np.zeros` function which declares a matrix of a given dimension and sets all elements to zero

# In[ ]:


n = 5
# define a matrix of dimension 10 x 10 and set all elements to zero
A = np.zeros( (n, n) )
print(A)


# In[ ]:


n = 5
# define a matrix of dimension 10 x 10 and set all elements to one
A = np.ones( (n, n) )
print(A)


# or as uniformly distributed random numbers on $[0,1]$

# In[ ]:


n = 4
# define a matrix of dimension 10 x 10 and set all elements to random numbers with x \in [0, 1]
A = np.random.rand(n, n)
print(A)


# The transpose of this matrix

# In[ ]:


A.T


# The dot product of two matrices can be computed with the `numpy.dot` function. Note that it is not the same as the arithmetic $*$ operation that performs elementwise multiplication

# In[ ]:


print(r'The dot product:')
AA = np.dot(A,A)
print(AA)
print(r'Element-wise product:')
print(A*A)


# The inverse of this matrix can be computed using the `numpy.linalg` module

# In[ ]:


Ainv = np.linalg.inv(A)
print(Ainv)


# The dot product of a matrix by its inverse returns the identity matrix (with small floating point errors). Verify that this is true:

# In[ ]:


np.dot(A,Ainv)


# The eigenvalues and eigenvectors of a matrix can be computed with the `eig` function

# In[ ]:


eigenvalues, eigenvectors = np.linalg.eig(A)
print('The eigenvalues are:\n',eigenvalues)
print('The eigenvectors are:\n',eigenvectors)


# ### Further examples with numpy
# 
# * The [NumPy tutorial](https://www.numpy.org/devdocs/user/quickstart.html) is a good resource.
# * You are also welcome to explore the more extensive numpy tutorial by Geron. It is reproduced, with the accompanying open-source Apache License, in the [handson-ml-notebooks](handson-ml-notebooks/tools_numpy.ipynb) directory.

# In[ ]:




