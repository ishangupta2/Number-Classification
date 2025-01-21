# README

# Setup

**You only need to do setup once**, then for future homeworks you can run `conda activate cse446`.

## Miniconda installation
Before you start working with this repo, you should install Anaconda.

**Before clicking in the link below** read notes below:

* Linux/Mac OS:
  * If using Linux/Mac please install command line version.
  * Make sure that you choose to initialize conda at startup.
    This will lead to fewer headaches in the future
* Windows:
  * If using Windows, we recommend using the Anaconda Terminal, which uses Bash-like syntax. 
* Low storage system
  * If you are low of storage (<10GB; for example attu), then Miniconda (see link below) might be a better option.

### Download links

[Anaconda (default)](https://www.anaconda.com/products/individual#Downloads)

[Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) (if running low on disk space)

You can find more detailed instructions for installation [at this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installing-conda-on-a-system-that-has-other-python-installations-or-packages).

## Environment Setup
First make sure you have at least ~5GB of free drive.
Then, from this directory, run:
```
conda env create -f environment.yaml
conda activate cse446
```

*Note*: The first command may take long time, especially if your connection is slow.

**Then, whenever you come back to work on homeworks,** just run: `conda activate cse446`.

![A quite long gif visualizing how to setup enviornment](./README_media/setup-env.gif)

### Optional - IDE setup

You are welcome to use any IDE as you would like.
Two popular for python are: VSCode (Microsoft) and PyCharm (JetBrains).

#### PyCharm
PyCharm is tailor made for Python, so the setup should be minimal.

Just make sure you point python interpreter to the right path for the project.
To do so, you should follow [these instructions](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html#add-existing-interpreter).

Optional: Please note, that to run linting (optional, see below) you will need to use terminal, and run `inv lint`.
You can try setting up linters through *External Programs* section.
If you succeed please let us know, and we will update this section for future reference :D

#### VSCode
VSCode is general purpose IDE, so you will need to install *Python* and *Pylance* extensions (both by Microsoft).

# Usage

## Testing
In this class we will use unittest framework in python to automatically grade coding problems.
Some of the tests are provided to you, so that you can validate your results.

To run tests:
```
inv test
```

The output should look something like this:
```
> inv test

FFF..
======================================================================
FAIL: test_polyfeatures_fives (public.poly_regression.test_poly_regression.TestPolyReg)
----------------------------------------------------------------------
Traceback (most recent call last):
  File ...
AssertionError: 
Arrays are not almost equal to 6 decimals

(shapes (1,), (20, 1) mismatch)
 x: array([1.])
 y: array([[5.],
       [5.],
       [5.],...
```

You can see that in the top there are 3 `F`'s and 2 `.`'s. `F`'s correspond to failed tests and `.` correspond to correct tests.

There are few things to note:

- Not all tests are equal. Some are worth more points. This will not be displayed when you run `inv test`.
- We **do not** provide you with all tests. There are many that hidden. Even if you pass all *public* tests you may still fail some *hidden* ones. We recommend submitting your code to Gradescope early and often so you can check the autograder score before the deadline, in case you would like to resubmit.

### Testing specific problem
Unfortunately the `unittest` framework doesn't allow for testing specific file.
However, you can run tests against specific problem, using the problem's directory name.
To do so run:
```
inv test --problem <problem-name>
```
For example:
```
> inv test --problem poly_regression

test_fit_and_predict_cubic (test_poly_regression.TestPolyReg) ... ok
test_fit_and_predict_straight_line (test_poly_regression.TestPolyReg) ... ok
test_fit_cubic (test_poly_regression.TestPolyReg) ... ok
test_fit_hard (test_poly_regression.TestPolyReg) ... ok
test_fit_linear (test_poly_regression.TestPolyReg) ... ok
test_fit_straight_line (test_poly_regression.TestPolyReg) ... ok
test_mean_squared_error (test_poly_regression.TestPolyReg) ... ok
test_polyfeatures_fives (test_poly_regression.TestPolyReg) ... ok
test_polyfeatures_ones (test_poly_regression.TestPolyReg) ... ok
test_polyfeatures_twos (test_poly_regression.TestPolyReg) ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.197s

OK
```
