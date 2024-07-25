from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(filename)-> List[str]:
    requirements=[]
    with open(filename) as f:
        requirements = f.readlines()
    requirements = [x.replace("\n","") for x in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name='UsedCarPricePrediction',
    version='0.1',
    author='Vibhore Jain',
    author_email='vjmj4005@gmail.com',
    description='Used Car Price Prediction',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    )