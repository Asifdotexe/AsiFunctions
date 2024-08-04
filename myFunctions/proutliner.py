import os
import argparse

def create_project_structure(project_name):
    # Define the directory structure
    directories = [
        f"{project_name}/data/raw",
        f"{project_name}/data/processed",
        f"{project_name}/code/notebooks/src",
        f"{project_name}/code/notebooks",
        f"{project_name}/code/scripts"
    ]

    # Define placeholder files
    files = [
        f"{project_name}/README.md",
        f"{project_name}/.gitignore",
        f"{project_name}/requirements.txt",
        f"{project_name}/code/notebooks/src/__init__.py",
        f"{project_name}/code/scripts/__init__.py"
    ]

    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Create placeholder files
    for file in files:
        with open(file, 'w') as f:
            pass

    # Add content to .gitignore
    gitignore_content = """\
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
# According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
# However, in case of collaboration, if having platform-specific dependencies or dependencies
# having no cross-platform support, pipenv may install dependencies that don't work, or not
# install all needed dependencies.
#Pipfile.lock

# PEP 582; used by e.g. github.com/David-OConnor/pyflow
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# dotenv
.env
.env.*

# virtualenv
venv/
ENV/
env/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/
"""

    with open(f"{project_name}/.gitignore", 'w') as f:
        f.write(gitignore_content)

    print(f"{project_name} project directory structure created successfully.")

def main():
    parser = argparse.ArgumentParser(description='Create a project directory structure.')
    parser.add_argument('project_name', type=str, help='The name or path of the project directory to create.')

    args = parser.parse_args()
    create_project_structure(args.project_name)

if __name__ == '__main__':
    main()