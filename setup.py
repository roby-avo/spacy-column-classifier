from setuptools import setup, find_packages

setup(
    name='column_classifier',
    version='0.1.1',
    description='A column classifier using spaCy for entity recognition.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Roberto',
    author_email='roberto.avogadro@sintef.no',
    url='https://github.com/roby-avo/spacy-column-classifier',
    license='Apache License 2.0',
    packages=find_packages(),  # Automatically finds your 'column_classifier' package
    install_requires=[
        'spacy',  # Add other dependencies here
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.7',
)