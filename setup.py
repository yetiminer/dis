from setuptools import setup, find_packages

setup(
    name = "secTools",
    author ="Henry Ashton",
    version = "0.1.0",
    packages = find_packages(exclude=['*test']),
    install_requires = ['argparse','pandas','numpy','os'],
    entry_points={
        'console_scripts': [
            'hunt = adventure.command:process'
        ]})
