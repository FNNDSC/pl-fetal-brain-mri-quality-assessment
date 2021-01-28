from os import path
from setuptools import setup

with open(path.join(path.dirname(path.abspath(__file__)), 'README.md')) as f:
    readme = f.read()

setup(
    name             = 'fetal_brain_quality_assessment',
    version          = '0.1',
    description      = 'Automatic fetal brain quality assessment tool.',
    long_description = readme,
    author           = 'Iván Legorreta',
    author_email     = 'ilegorreta@outlook.com',
    url              = 'https://github.com/FNNDSC/Automatic-Fetal-Brain-Quality-Assessment-Tool',
    packages         = ['fetal_brain_quality_assessment'],
    install_requires = ['chrisapp'],
    license          = 'MIT',
    zip_safe         = False,
    python_requires  = '>=3.8',
    entry_points     = {
        'console_scripts': [
            'fetal_brain_quality_assessment = fetal_brain_quality_assessment.__main__:main'
            ]
        }
)
