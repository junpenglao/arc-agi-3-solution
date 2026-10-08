.PHONY: test

test:
	cd .. && uv --quiet --project ARC3-Inference run --locked --extra dev pytest \
		-c ARC3-Inference/pyproject.toml -q \
		serving/franzen_h200_test.py serving/qsa_probe_test.py
