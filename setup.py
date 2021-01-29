from os import path
from setuptools import setup

with open(path.join(path.dirname(path.abspath(__file__)), 'README.md')) as f:
    readme = f.read()

setup(
    name             = 'fetal_brain_quality_assessment',
    version          = '1.0.0',
    description      = 'Automatic fetal brain quality assessment tool',
    long_description = readme,
    author           = 'Iván Legorreta',
    author_email     = 'ilegorreta@outlook.com',
    url              = 'https://github.com/FNNDSC/pl-fetal-brain-mri-quality-assessment',
    packages         = ['fetal_brain_quality_assessment'],
    install_requires = ['chrisapp', 'colorlog'],
    license          = 'MIT',
    zip_safe         = False,
    python_requires  = '>=3.8',
    entry_points     = {
        'console_scripts': [
            'fetal_brain_quality_assessment = fetal_brain_quality_assessment.__main__:main'
            ]
        }
)
