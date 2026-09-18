# How to Run

## Setup

```bash
git clone https://github.com/ChirayuMarathe/LinkedIn_Estimator.git
cd LinkedIn_Estimator
pip install -r requirements.txt
```

## Train the Model

```bash
python scripts/train.py
```

## Run the Web App

```bash
python -m uvicorn demo.app:app --reload
```

Open http://127.0.0.1:8000

## Run from Command Line

```bash
python demo/cli.py --company ubiquiti --format video "Your post text here..."
```

## Run Tests

```bash
python -m pytest tests/ -v
```
