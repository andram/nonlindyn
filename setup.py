from setuptools import setup
import site
import sys
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

setup(name='nonlindyn',
      version='0.2',
      description='Library to support analysing nonlinear dynamical systems',
      url='http://github.com/andram/nonlindyn',
      author='Andreas Amann',
      author_email='a.amann@ucc.ie',
      license='GPL',
      packages=['nonlindyn'],
)
