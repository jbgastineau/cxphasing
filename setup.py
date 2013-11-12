from distutils.core import setup

setup(
    name='CXPhasing',
    version='0.5.0',
    author='D. J. Vine',
    author_email='djvine@gmail.com',
    packages=['cxphasing', 'cxphasing.cxparams'],
    scripts=[],
    url='http://pypi.python.org/pypi/CXPhasing/',
    license='LICENSE.txt',
    description='Coherent X-ray ptychography phase retrieval',
    long_description=open('README.txt').read(),
    requires=[
        "numpy (>=1.7)",
        "scipy",
        "pylab",
        "Image",
        "h5py",
        "libtiff",
    ],
)
