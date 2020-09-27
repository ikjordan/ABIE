from setuptools import setup, find_packages
from distutils.core import setup, Extension
import sys
import sysconfig
# name is the package name in pip, should be lower case and not conflict with existing packages
# packages are code source


suffix = sysconfig.get_config_var('EXT_SUFFIX')
if suffix is None:
    suffix = ".so"

if sys.platform == 'darwin':
    from distutils import sysconfig
    vars = sysconfig.get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-shared')
    extra_link_args=['-Wl,-fcommon,-install_name,@rpath/libabie'+suffix]
else:
    extra_link_args=['-fcommon']

module_abie = Extension('libabie',
                        sources = ['libabie/integrator_gauss_radau15.c',
                            'libabie/integrator_wisdom_holman.c',
                            'libabie/integrator_runge_kutta.c',
                            'libabie/common.c',
                            'libabie/additional_forces.c'],
                        include_dirs = ['libabie'],
                        extra_compile_args=['-fstrict-aliasing', '-O3','-std=c99','-march=native','-fPIC', '-shared', '-fcommon'],
                        extra_link_args=extra_link_args,
                        )

# Note on Raspberry pi installation of toml, numpy and h5py may fail at 
# comilation stage. If so then install these packages manually, then 
# comment out the "install_requires" line below
setup(name='astroabie',
      version='0.2',
      description='Alice-Bob Integrator Environment',
      url='https://github.com/maxwelltsai/MPA',
      author='Maxwell Cai, Javier Roa, Adrian Hamers, Nathan Leigh',
      author_email='maxwellemail@gmail.com',
      license='BSD 2-Clause',
      packages=find_packages(),
      zip_safe=False,
      install_requires=['toml', 'numpy', 'h5py'],
      entry_points={'console_scripts': ['abie = ABIE.abie:main'] },
      ext_modules = [module_abie]
      )
