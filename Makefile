.PHONY: check

check:
	python -m compileall -q scripts
	PYTHONPATH=scripts/smal_pets python scripts/smal_pets/check_gaussians.py
	PYTHONPATH=scripts/smal_pets python scripts/smal_pets/check_deformation.py /tmp/gaussian-dog-deformation.json
