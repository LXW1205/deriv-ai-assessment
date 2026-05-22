.PHONY: run validate api clean install

PYTHON := python3

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) main.py

validate:
	$(PYTHON) validate.py

api:
	$(PYTHON) api.py

clean:
	rm -rf artifacts/
	rm -f llm_calls.jsonl
