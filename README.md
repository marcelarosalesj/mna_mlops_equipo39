# Students Grade Prediction

## Roles

- David Gutierrez - Subject matter expert
- Raul Rodriguez - Data Scientist
- Victor Soto - Data Engineer
- Marcela Rosales - Software Engineer
- Roberto Ortega - MLOps Engineer

# Setup project

```
git clone git@github.com:marcelarosalesj/mna_mlops_equipo39.git

cd mna_mlops_equipo39

# git checkout refactor # make sure you're on this branch

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

export PYTHONPATH=$PYTHONPATH:$(pwd)

mkdir artifacts

dvc repro
```

# Start mlflow server

```
mlflow server --host 127.0.0.1 --port 8080
```

# Unit tests and Integration tests

```
pytest -v tests/
```
