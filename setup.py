from setuptools import setup, find_packages

with open('requirements.txt', 'r', encoding='utf-8') as req:
    requirements = [line.strip() for line in req if line.strip() and not line.startswith('#')]

setup(
    name='titanic_survival_project',
    version='1.0.0',
    author='Group 2 - IIUI',
    description='Titanic Survival Prediction Project',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'titanic-pipeline=run_pipeline:main'
        ]
    }
)
