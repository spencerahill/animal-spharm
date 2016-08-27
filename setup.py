import setuptools

setuptools.setup(
    name="animal_spharm",
    version="0.0",
    packages=setuptools.find_packages(),
    author="Spencer A. Hill",
    author_email="spencerahill@gmail.com",
    description="Xarray-based wrapper around pyspharm and windspharm packages",
    license="Apache",
    keywords="climate science",
    url="https://github.com/spencerahill/animal-spharm",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Atmospheric Science'
    ]
)
