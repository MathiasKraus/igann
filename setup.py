from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='igann',
    version='0.1.5',
    author='Mathias Kraus',
    author_email='mathias.sebastian.kraus@gmail.com',
    description='Implementation of Interpretable Generalized Additive Neural Networks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MathiasKraus/igann',
    license='MIT',
    packages=['igann'],
    zip_safe=False,
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'torch>=1.9.0',
        'matplotlib',
        'seaborn',
    ]
)

