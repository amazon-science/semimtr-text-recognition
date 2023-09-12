import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="semimtr-text-recognition",
    version="0.0.1",
    author="SemiMTR Text Recognition",
    author_email="aws-cv-text-ocr@amazon.com",
    description="This package contains the package for SemiMTR",
    long_description_content_type="text/markdown",
    url="https://github.com/amazon-science/semimtr-text-recognition.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Amazon Copyright",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
