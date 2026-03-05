SHELL  := /bin/bash
.PHONY: serve eval perf guardrails improve clean

VENV   := venv/bin/activate
PYTHON := source $(VENV) && python

serve:
	$(PYTHON) serve/serve.py

eval:
	$(PYTHON) eval_runner/run_eval.py

eval-quick:
	$(PYTHON) eval_runner/run_eval.py --limit 5

perf:
	$(PYTHON) perf/load_test.py
	jupyter nbconvert --to notebook --execute perf/analysis.ipynb --output analysis.ipynb

guardrails:
	$(PYTHON) guardrails/validate.py

guardrails-quick:
	$(PYTHON) guardrails/validate.py --skip-harness

improve:
	bash improve/eval.sh

improve-prepare:
	bash improve/eval.sh prepare

improve-optimize:
	bash improve/eval.sh optimize

improve-infer:
	bash improve/eval.sh infer

test:
	source $(VENV) && pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf eval_runner/cache/
	rm -rf improve/results/
	rm -rf improve/tasks/
