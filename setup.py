try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'An AO registration app based on MATLAB code by S.Burns',
    'author': 'Tom Wright',
    'url': 'https://github.com/tomwright01/AO_Registration',
    'download_url': 'https://github.com/tomwright01/AO_Registration',
    'author_email': 'tom@maladmin.com',
    'version': '0.1',
    'install_requires': ['nose','numpy','scipy'],
    'py_modules': ['AoRegistration','StackTools','ImageTools','tests'],
    'scripts': [],
    'name': 'AoRegistration'
}

setup(**config)
