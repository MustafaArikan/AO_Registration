try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'An AO registration app based on MATLAB code by S.Burns',
    'author': 'Tom Wright',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it.',
    'author_email': 'tom@maladmin.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['AoRegistration'],
    'scripts': [],
    'name': 'AoRegistration'
}

setup(**config)