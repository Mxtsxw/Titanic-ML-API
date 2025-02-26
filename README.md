# Titanic ML API

This project is a simple API that predicts the survival of passengers on the Titanic. It uses a machine learning model that was trained on the Titanic dataset from Kaggle.

## Repository Structure

```
.
├── artifcats/
│   └── pipeline.pkl
├── notebooks/
│   ├── EDA.ipynb
│   └── ML.ipynb
├── scripts/
│   └── utils.py
├── serving/
│   └── api.py
└── webapp/
    └── app.py
```

### Artifacts

The `artifacts` directory contains the serialized pipeline that was trained on the Titanic dataset.

### Notebooks

The `notebooks` directory contains the Jupyter notebooks that were used to explore the Titanic dataset and train the machine learning model.

### Scripts

The `scripts` directory contains utility functions that are used to preprocess the data and make predictions.

### Serving

The `serving` directory contains the API that serves the machine learning model.

## Usage

To run the API, execute the following command from the `serving` directory:

```bash
uvicorn api:app --host 127.0.0.1 --port 8080
```

The API will be available at `http://localhost:8000`.

## Endpoints

`/predict` **POST**

**Request**

```json
{
  "PassengerId": 1,
  "Pclass": 1,
  "Name": "John Doe",
  "Sex": "female",
  "Age": 22,
  "SibSp": 0,
  "Parch": 0,
  "Ticket": "A/5 21171",
  "Fare": 22,
  "Cabin": null,
  "Embarked": "S"
}
```

**Response**

```json
{
  "Survived": 1
}
```

## Pipeline explanation 

The pipeline is composed of the following steps:
- **Feature Selection** : The features used in the model are `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.
- **Preprocessing**: The data is preprocessed by filling missing values and encoding categorical variables.
- **Model**: The model used is a Random Forest Classifier.
- **Prediction**: The model predicts the survival of passengers on the Titanic.

Here is a diagram of the pipeline:

<img alt="pipeline.png" src="misc/pipeline.png" width="600"/>

## Titanic - Machine Learning from Disaster Kaggle Competition

A notebook and submission file for the Titanic Kaggle competition can be found in the `notebooks` and `data` directory.

## Streamlit Web App

A Streamlit web app that allows users to interact with the API and infer predictions can be found in the `webapp` directory.

<img alt="webapp.png" src="misc/webapp.png" width="600"/>

To run the app, execute the following command from the `webapp` directory:

```bash
streamlit run app.py
```

The API will be available at `http://localhost:8501`.

## Future Works 

- [ ] Add a Dockerfile to containerize the API and App
