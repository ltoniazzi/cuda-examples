from setuptools import setup, find_packages

setup(
    name='backprop',  # Use your actual package name, NOT 'src'
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A simple description of your package',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
