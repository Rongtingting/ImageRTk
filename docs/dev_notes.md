

```shell
cd ~/Github_repos/imagertk
poetry init

poetry add numpy scikit-image tifffile matplotlib
poetry add --group dev pytest black ruff mypy

poetry add pandas


poetry add git+https://github.com/wuwenrui555/pycodex.git@dev
(imagertk) rongtinghuang@darthvader:~/Github_repos/ImageRTk$ poetry add git+https://github.com/wuwenrui555/pycodex.git@dev

Updating dependencies
Resolving dependencies... (9.5s)

The current project's supported Python range (>=3.11) is not compatible with some of the required packages Python requirement:
  - pycodex requires Python >=3.9, <3.11, so it will not be installable for Python >=3.11

Because imagertk depends on pycodex (0.2.3) @ git+https://github.com/wuwenrui555/pycodex.git@dev which requires Python >=3.9, <3.11, version solving failed.

  * Check your dependencies Python requirement: The Python requirement can be specified via the `python` or `markers` properties

    For pycodex, a possible solution would be to set the `python` property to "<empty>"

    https://python-poetry.org/docs/dependency-specification/#python-restricted-dependencies,
    https://python-poetry.org/docs/dependency-specification/#using-environment-markers


poetry add deepcell





# Use Poetry Commands Within Your Conda Env
poetry install
poetry run pytest
poetry run black imagertk
poetry run ruff check imagertk




```

# Change to PYTHON 3.10
DEEPCELL REQUIREMENTS IS STRICT

```shell

poetry add deepcell
poetry add git+https://github.com/wuwenrui555/pycodex.git@dev

(imagertk-py310) rongtinghuang@darthvader:~/Github_repos/ImageRTk$ poetry add git+https://github.com/wuwenrui555/pycodex.git@dev

Updating dependencies

Resolving dependencies... (5.5s)
Because pycodex (0.2.3) @ git+https://github.com/wuwenrui555/pycodex.git@dev depends on deepcell (<=0.12.6)
 and imagertk depends on deepcell (>=0.12.10,<0.13.0), pycodex is forbidden.
So, because imagertk depends on pycodex (0.2.3) @ git+https://github.com/wuwenrui555/pycodex.git@dev, version solving failed.



```
