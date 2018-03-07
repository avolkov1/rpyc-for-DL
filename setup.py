'''
'''
import sys
from setuptools import setup, find_packages

IS_PY_2 = (sys.version_info[0] <= 2)

_pkgname = 'rpyc_DL'
packages_list = [subpkg for subpkg in find_packages(where=_pkgname)]

# print('PACKAGES LIST: {}'.format(packages_list))

install_requires = [
    'nvidia-ml-py>=7.352.0' if IS_PY_2 else 'nvidia-ml-py3>=7.352.0',
    'rpyc',
    'dill',
    'fasteners'
]

setup(
    name='rpycdl_lib',
    version='1.0',
    description='RPyC Deep Learning Services Demo',
    author='Alex Volkov',
    url='https://github.com/avolkov1/rpyc-for-DL',
    packages=packages_list,
    # packages=['rpycdl_lib'],
    package_dir={'': 'rpyc_DL'},
    install_requires=install_requires,
    license='unlicense.org',
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Unlicense',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX :: Linux',
    ]
)
