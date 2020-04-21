import setuptools

setuptools.setup(
    name="metalibm", # Replace with your own username
    version="1.0.0",
    author="Nicolas Brunie",
    author_email="nbrunie@kalray.eu",
    description="a math kernel code generation",
    url="https://github.com/kalray/metalibm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "sollya @ git+https://gitlab.com/metalibm-dev/pythonsollya",
    ],
    python_requires='>=3.5',
)
