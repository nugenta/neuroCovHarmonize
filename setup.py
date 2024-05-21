from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='neuroCovHarmonize',
      version='2.4.5',
      description='Harmonization tools for multi-center neuroimaging studies.',
      long_description=readme(),
      url='https://github.com/nugenta/neuroCovHarmonize.git',
      author='Raymond Pomponio and Allison Nugent',
      author_email='raymond.pomponio@outlook.edu',
      license='MIT',
      packages=['neuroCovHarmonize'],
      install_requires=['numpy', 'pandas', 'nibabel', 'statsmodels>=0.12.0', 'neuroCombat==0.2.12'],
      zip_safe=False)
