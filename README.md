# Fork of Moving Planets Around (MPA) Project
I have been writing toy n-body simulations for many years.  
I found reading *Moving Planets Around* to be a great way to increase my knowledge of the topic.  

This fork aims to build 
on the official code repository for the book. The following changes have been made:

1. Address issues in some of the integrators that prevented them running. *done*
2. Address some issues I found with Python package management *done*
3. Add support for MS Windows, using Microsoft Visual Studio Community 2019 
(free to non-commercial users) *done*
4. Support MS Windows builds using gcc under msys2 (minGW 64 bit) *done*
5. Add extra example files *several added* see [Examples](#Examples)
6. Get the CUDA code working under Windows *Done*
7. Fix the use of local memory (tiling) in CUDA *Done*
8. Implement OpenCL acceleration *Done for AMD, Intel and NVidia under Windows and Linux*
9. Give option of using OpenMP on multi-core targets *Done for Windows and Linux*
10. Make changes to `snapshot_serialization.py` to support windows, and allow it to be called 
from Python programatically *done*

## Notes
1. It appears that major changes were made to the code to introduce the accelerated C integrators. This broke
some of the older integrator inplementations in Python. These issues have now been addressed, and
all integrators should now work in both `ctypes` (where implemented) and `numpy` mode
2. Issues surrounding division that broke execution when using Python 3 have been corrected
3. Some C arrays were declared on the stack. This caused stack overflows with large particle simulations. 
Heap memory is now used instead
4. Addressed issues with timestep and energy delta calculations in the `Wisdom-Holman` integrator. 
Accelerated the calculation of energy delta by adding a C function called via `ctypes`. This function is also used when displaying energy delta graphs created from `.h5` files
5. The package configuration has been modified, so that code can be called in the same way, regardless
of whether the `astroabie` package has been installed. Examples have been moved into a separate
directory
6. Added an option to control the amount of data written to the console during a run
7. Loading the shared library from the installed package has been refined, to allow for instances where
the library name is decorated by the architecture during installation
8. A MS Visual Studio 2019 solution is provided. This creates a DLL file from the C code, and allows easy 
access to the debugger for both the Python and C code
9. A CUDA MSVC 2019 solution is also provided. Adding a CUDA aware build option to MSVC precludes that solution
being loaded on a system where CUDA is not installed. Therefore two solutions are provided `ABIE` and
`ABIE_CUDA`. CUDA is enabled in `ABIE_CUDA` when the `GPU` preprocessor directive is added
10. The CUDA version that uses shared local memory, which was removed from the original repository, has been fixed, and is now the default. The global memory version from the repository can be re-enabled via a compile flag. Note that the CUDA implementation has only been tested under Windows
11. A third MSVC solution supports OpenCL (`ABIE_OpenCL`). The OpenCL version uses shared local memory, and executes in similar speed to the shared memory CUDA implementation on my PC. It has also been tested on Intel and AMD processors, and under Linux. For reference, an OpenCL global memory version is also provided and can be enabled via a compile flag. OpenCL is enabled when the `OPENCL` preprocessor directive is added
12. ABIE uses C Variable Length Arrays (VLA), which are not supported by the MSVC compiler (MSVC is not
completely C99 compliant). Therefore the LLVM (`clang-cl`) tool chain is used with Visual Studio 2019
13. The `clang-cl` compiler does support OpenMP, but the required DLL is not installed automatically. `libomp.dll`
can be found in the LLVM section of the MSVC 2019 installation. Either copy it to the DLL directory for your system,
or paste a copy into the same directory as `libabie.dll`. OpenMP is enabled when the `OPENMP` preprocessor directive is added
14. OpenMP is enabled by default in the libabie `Makefile`
15. The MSVC solutions are configured so that the compiler generates AVX instructions. On older processors that do not
support AVX, this can result in `error 0xc000001d` at startup. If this occurs modify the MSVC solution so that AVX instructions are not generated 
16. The code has been tested under MS Windows 10, MinGW 64-bit running on Windows 10, Ubuntu 20.04 running
in a VirtualBox VM and native, and a 4GB Raspberry PI 4 running Raspberry Pi OS
17. I do not have acccess to an Apple Mac, so I may have inadvertantly broken support for that platform
18. To make intellisense work correctly for C99 code with `Clang` an additional option of `-std=c99` is set. 
This is valid in `Clang`, and triggers intellisense to work correctly, but is ignored in `Clang-cl`, and so
causes a warning to be generated
19. Clang supports 80 bit `long double`, but in Clang-cl `long double` is set to 64 bits for 
compatibility with MSVC. Use MinGW64 to build a DLL that will provide 80 bit `long double` support under Windows
20. The C files in the project are built into a DLL. I failed to configure `setup.py` to automatically 
trigger a build using `Clang`, so before running `setup.py` `libabie.dll` should be built in Visual Studio.
The DLL will then be packaged correctly
21. Under MinGW 64bit the DLL (`libabie.dll`) can be built by invoking `make`. This should be done before 
running `setup.py`. The DLL will then be packaged correctly
22. Note that there appears to be an issue in on the Raspberry Pi, where specifying dependencies can cause 
`Cython` to fail. The workaround is to comment the dependencies out from `setup.py` and install them
manually
23. The `particles` class has been extended to give an option to position the COM of the system at the origin 
with zero velocity
24. If, after installing astroabie using:  
`python setup.py install`  
you want to remove it, then use the following:  
`pip uninstall astroabie`  
To trigger a full rebuild of `libabie` when using `setup.py`, the `build` directory must also be purged

### MinGW 64bit - h5py issue
Currently (September 2020) there is a known issue with h5py under msys2 minGW64. 
For example bit see:  
https://github.com/msys2/MINGW-packages/issues/6612

To correct for this:
1. Remove the h5py package (if installed)  
`pacman -R mingw-w64-x86_64-python-h5py`

2. Remove the hdf5 package (if installed)  
`pacman -R hdf5`

3. Download an old version of hdf5 from 
http://repo.msys2.org/mingw/x86_64/  

4. Install it  
`pacman -U mingw-w64-x86_64-hdf5-1.8.21-2-any.pkg.tar.xz`

5. Lock the version, so that it is not updated  
Add the line:  
`IgnorePkg   = mingw-w64-x86_64-hdf5-1.8.21-2`  
To `c:\msys64\etc\pacman.conf`

6. Rebuild h5py  
Download source file from  
http://repo.msys2.org/mingw/sources/  
untar
`tar -xf  mingw-w64-python-h5py-2.10.0-1.src.tar.gz`

7. Build it by going to the untared directory 
`MINGW_INSTALLS=mingw64 makepkg-mingw -sLf`

8. Install it  
`pacman -U mingw-w64-x86_64-python-h5py-2.10.0-1-any.pkg.tar.zst`


## Examples
The example files will use the `astroabie` package, if it is installed. Otherwise they will fall back to
use the local `ABIE` module files


### `run.py`
The original example supplied in the base repository, extended to call a modified
`snapshot_serialization.py` to convert the `h5` data file and then use `Matplotlib` to plot the results in 3d

### `simple.py`
Five simple 2 to 5 body cases, some of which become unstable over time. Uses `Matplotlib` to plot the 
results in 2d

### `solar.py`
Uses the solar system position and velocity information provided in *Moving Planets Around* to 
simulate the solar system. Uses `Matplotlib`
to plot the results in 2d.  
By default it plots the solar system for 1000 years. By setting the variable `million`
to `True` the simulation will run for 1 millions years and 
recreate fig. 9.8 from the book. The figure is optionally smoothed
using a low pass filter.   
The energy delta over time is displayed. If necessary coordinates
are converted to Baryocentric, so that the energy calculation is consistent over time

### `saturn.py`
Simulates two moons of Saturn - Epimetheus and Janus, which are co-orbital. The change in
orbital radius every 4 to 5 years can be seen as the moons approach each other. As they get close the semimajor axis of the 
trailing moon increases, until it is larger
then that of the leading moon. At that point is falls further behind the leading moon.
This can be seen in the graph of the relative angle between the bodies, which never goes to 0. 
 
### `oort.py`
Based on the example in chapter 10.

### `kuiper.py`
A simulation comprising the Sun, Neptune and a large number of small Kuiper belt like objects. The simulation shows some
of the the objects falling into resonance with Neptune. See e.g. https://en.wikipedia.org/wiki/Resonant_trans-Neptunian_object

Illustrates the use of CUDA, OpenCL and OpenMP in more demanding simulations. Switch from Wisdom-Holman to Gauss-Radau15 to enable CUDA or OpenCL

#### `display.py`
Helper class to display multiple graphs using `matplotlib`  

#### `h5.py`
Helper class to perform transforms on the data within the `h5` file

#### `tools.py`
Added helper methods to calculate residual momentum in a system, and to convert
between coordinate systems  

### The original README follows:

# Moving Planets Around (MPA) Project

*Moving Planets Around* is an education book project that teaches students to build a state-of-the-art N-Body code for planetary system dynamics from the stretch. The code built throughout the storyline of the book is hosted here. The book will be published by the MIT Press around September 2020. Stay tuned! See: https://mitpress.mit.edu/books/moving-planets-around 

### The Alice-Bob Integrator Environment (ABIE)
------

The source code of `ABIE` is stored under the directory ABIE. From this particular directory, ABIE can be launched with

    python abie.py -c ../Test_ABIE/solar_system.conf -o output.hdf5
  
Where `../Test_ABIE/solar_system.conf` is the path and file name of the initial condition config file. `output.hdf5` is the file name of integration data output.

ABIE also support loading initial conditions from Rebound data file (requires Rebound). For example, one can use ABIE to evolve a system with Kozai-Lidov generated by Rebound:

    python abie.py ../Test_ABIE/kozai_lidov.reb -t <t_end>
    
The argument `t_end` is needed because Rebound doesn't store the termination time in its data files. 

The unit of `t_end` depends on the setting of `CONST_G` parameter.

### Using ABIE as a library

`ABIE` can also be executed programatically. You could install `ABIE` as a python package:

    python setup.py install

Note that `setup.py` is in the parent directory of `ABIE`'s main source code directory. See an example in the `run.py` file.

### ABIE output format

`ABIE` uses the HDF5 format to store its integration output data. The internal layout of the HDF5 file looks like this:

    /Step#0
        ecc: eccentricities of each object (1 - N) as a function of time (0, t-1): [t * N] array
        inc: inclinations of each object (1 - N) as a function of time (0, t-1): [t * N] array
        hash: unique integer identifier for each particle: [t * N] array
        mass: masses of each planet: [t * N] array
        ptype: particle type of each object. 0 = massive particles; 1 = test particles; 2 = low-mass particles. [t * N] array
        radius: radius of each object. [t * N] array
        semi: semi-major axis of each object. [t * N] array
        time: the time vector. [t * 1] vector
        vx: [t * N]
        vy: [t * N]
        vz: [t * N]
        x: [t * N]
        y: [t * N]
        z: [t * N]
    /Step#1
        ...
    /Step#2
        ...
    ...
    /Step#n
    
For efficient output, ABIE maintains a buffer to temporarily store its integration data. The default buffer length is set to 1024, which means that it will accumulate 1024 snapshot output until it flushes the data into the HDF5 file and creates a `Step#n` group. The resulting HDF5 datasets can be loaded easily using the `h5py` package:

```python
import h5py
h5f = h5py.File('data.hdf5', 'r')
semi = h5f['/Step#0/semi'].value
ecc = h5f['/Step#0/ecc'].value
...
h5f.close()
```    

Sometimes, it is more elegant to get rid of the `Step#n` data structure in the HDF5 file (i.e. combine `Step#0`, `Step#1`, ..., `Step#n` into flatten arrays. The `ABIE` package contains a tool to seralize the snapshot. For example, suppose that `ABIE` generates a data file `data.hdf5` contains the `Step#n` structure, the following command

```
python snapshot_serialization -f data.hdf5
```
will generate a flattened file called `data` (still in hdf5 format). In this case, the data can be accessed in this way:
```python
import h5py
h5f = h5py.File('data.hdf5', 'r')
semi = h5f['/semi'].value  # gets the semi-axis array for the entire simulation
ecc = h5f['/ecc'].value    # gets the eccentricity array for the entire simulation
...
h5f.close()
```    

### Integrators and Computational Acceleration

`ABIE` implements all its integrators in both Python (for educational purpose) and in C (for performance). The currently supported integrators are:

- Forward Euler integrator
- Leapfrog
- Adams-Bashforth
- Runge-Kutta
- Gauss-Radau15 *(default)*
- Wisdom-Holman

By default, ABIE will execute the C implementation of the Gauss-Radau15 integrator. This integrator is well-optimized and preserves energy to ~ 10^{-15}. To change the integrator and use the python implementation, one could either edit the config file:

     [integration]
     integrator = 'GaussRadau15'
     # integrator = 'RungeKutta'
     # integrator = 'LeapFrog'
     tf = 356
     h = 0.1  # optional for certain integrators
     t0 = 0.0
     acc_method = 'ctypes'
     # acc_method = 'numpy'

Or use the Python script:

```python
from ABIE.abie import ABIE
sim = ABIE()
sim.CONST_G = 1.0
sim.integrator = 'GaussRadau15'
```
    
### Improve the precision of ABIE

By default, `ABIE` uses double precision. For some special cases (for example, integrating a Kozai-Lidov system where the eccentricity can be very high), the precision of the integrator can be adjusted by simply changing the following lines in `Makefile` from

```Makefile
LONGDOUBLE = 0
```
    
to 

```Makefile
LONGDOUBLE = 1
```

And run `make clean; make` again. This  will causes the integrator to use the [`long double`](https://en.wikipedia.org/wiki/Long_double) data type. When using the Gauss-Radau15 integrator, the energy conservation can be better than 10^{-16} in this case (shown as `dE/E = 0` in the terminal), even after evolving the system through multiple Kozai cycles. This, however, will takes about 2x time to finish evolving the same system.


### Accelerate ABIE using CUDA/GPU

`ABIE` supports GPU acceleration. For large `N` systems (N>512), using the GPU could result in substential speed up. To enable the GPU support, modify the `Makefile` from

```Makefile
GPU = 0
```
    
to 

```Makefile
GPU = 1
```

And then recompile the code. Note that it is not recommended to use GPU for small N systems.

  


