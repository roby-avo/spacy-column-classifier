from setuptools import setup, find_packages

setup(
    name='column_classifier',  # This is the actual package name
    version='0.1.0',
    description='A column classifier using spaCy for entity recognition.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Roberto',
    author_email='roberto.avogadro@sintef.no',
    url='https://github.com/roby-avo/spacy-column-classifier',  # Keep your GitHub URL
    license='Apache License 2.0',
    packages=find_packages(),  # Automatically finds the 'column_classifier' package
    install_requires=[
        'spacy',  # Add other dependencies here
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.7',
)