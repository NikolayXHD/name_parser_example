clear:
	rm -rf ./.mypy_cache ./.pytest_cache
	find . -type f -name "*.pyc" -print0 | xargs -r0 rm

format:
	pipenv run python -m brunette \
      --single-quotes \
      --target-version py38 \
      --line-length 79 \
      .

lint:
	pipenv run python -m flake8
	pipenv run python -m mypy notebooks tests

test:
	pipenv run python -m pytest -l tests

# input: make test-v
# result: pipenv run python -m pytest -lv tests
test-%:
	pipenv run python -m pytest -l$* tests

test-failed:
	pipenv run python -m pytest -l --last-failed tests

check: format lint test

jupyter:
	pipenv run jupyter notebook
