from distutils.core import setup

setup(
    name='Cochlear',
    version='0.1',
    author='Brad Buran (bburan@alum.mit.edu)',
    packages=['cochlear'],
    url='http://github.com/bburan/cochlear',
    license='LICENSE.txt',
    description='Module for various auditory experiments',
    requires=['numpy'],
    scripts=[
        'scripts/merge_files.py',
        'scripts/launcher.py',
    ],
)
