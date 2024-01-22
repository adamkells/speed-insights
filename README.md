# speed-insights

This repository hosts a tool that simplifies and accelerates the model comparison process. 🔄

How It Works:

1. Provide Your Dataset: Input your dataset.
2. Bring Your Models: Include your pre-trained models.
3. Automated Insights: Easily generate metrics, visualizations, and insights to compare model performances.

This tool is designed to make model evaluation straightforward and efficient. 📈🤖

## Origin Story

The actual reason for building this repo was because I mostly work on regression tasks and constantly find myself frustrated with the kaggle-ization of work.
I find people often pick a metric like RMSE at random and then go all in on optimizing it. I like to view a range of metrics (rather than just one) and I often implement methods that examine the performance without aggregating. I was inspired particularly by Vincent Warmerdams [DoubtLab](https://github.com/koaning/doubtlab).

## Features

- Automated model comparison.
- Computes a variety of standard metrics.
- Easily extensible to include extra metrics.
- Returns data rows for deeper investigation.
- Automatically generates and saves wide range of useful visualisations.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)

## Installation

To install the `speed-insights` package, you can use [Poetry](https://python-poetry.org/), a dependency management tool for Python.

1. Make sure you have Poetry installed. If not, you can install it by following the [official Poetry installation guide](https://python-poetry.org/docs/#installation).

2. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/adamkells/speed-insights.git
    ```

3. Navigate to the project directory:

    ```bash
    cd speed-insights
    ```

4. Install the project dependencies using Poetry:

    ```bash
    poetry install
    ```

    This will create a virtual environment and install all the required dependencies specified in the `pyproject.toml` file.

5. Activate the virtual environment:

    ```bash
    poetry shell
    ```

    This will activate the virtual environment and allow you to run the `speed-insights` commands.

6. You're all set! You can now start using the `speed-insights` package.

    ```python
    from speed_insights import SpeedInsights
    ```

If you encounter any issues during the installation process, please refer to the [Poetry documentation](https://python-poetry.org/docs/) or raise an issue in the [speed-insights repository](https://github.com/adamkells/speed-insights/issues).


## Usage

SpeedInsights is a designed to simplify and streamline the process of model evaluation. It includes a range of features that make it easy to compare the performance of different models, compute various standard metrics, and generate insightful visualizations.

With SpeedInsights, you can input your dataset, include pre-trained models, and generate metrics and insights to assess and compare model performances. At present it is specifically geared towards regression tasks but will be expanding in Q1 2024.

## Example

In the following example, we load a dataframe and train an sklearn fit/predict model as per the usual ML workflow. We then create a SpeedInsights object from X and y data as well as dictionary of my model.

```python
import pandas as pd

from speed_insights import SpeedInsights
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('my_data.csv')
y = data['target']
X = data.drop('target', axis=1)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the LinearRegression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Create an instance of the SpeedInsights class
insights = SpeedInsights(X_test, y_test, {'linear-model': model})

# Generate metrics, visualizations, and insights
insights.generate_metrics()
```

If we want to compare a range of models we can simply extend our dictionary:

```python
model_1 = LinearRegression()
model_2 = RandomForest()

model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)

insights = SpeedInsights(X_test, y_test, {'linear-model': model_1, 'rf-model': model_2})
insights.generate_metrics()
```

We can also generate visualisations of our predictions vs our ground truth with the generate_prediction_visualisations method.

```python
insights = SpeedInsights(X_test, y_test, {'linear-model': model_1, 'rf-model': model_2})
insights.generate_prediction_visualisations('output_folder')
```

## Roadmap
The following items are on the roadmap for Q1 2024.
- Expanding the list of metrics (SMAPE) as well as some pragmatic types of errors (largest error, std of error distribution).
- Adding a method to return potentially interesting rows for inspection.
- Expanding the visualisations (distribution of errors)
- Fairness investigation (groupwise inspection of metrics and errors)


## Contributing

Contributions are more than welcome! Feel free to raise an issue if you have any suggestions for additional visualisations and metrics.
