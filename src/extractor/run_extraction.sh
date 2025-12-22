# Run from project root
if [ -d ".venv" ]; then source .venv/bin/activate; else echo ".venv not found"; fi
python -m src.extractor.process_efcamdat
python -m src.extractor.process_write_improve
python -m src.extractor.process_icnale
python -m src.extractor.process_asag

#python -m src.extractor.combine
#python -m src.utils.splitter
