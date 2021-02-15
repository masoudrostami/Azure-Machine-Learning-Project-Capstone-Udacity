# Capstone - Udacity Machine Learning Engineer with Microsoft Azure

## Table of content
* [Overview](#overview)
* [Dataset](#dataset)
* [Project Set Up and Installation](#project-set-up-and-installation)
* [Automated ML](#automated-ml)
* [Hyperparameter Tuning](#hyperparameter-tuning)
* [Model Deployment](#model-deployment)
* [Screen Recording](#screen-recording)

---

## Overview

In this project, we will be using Azure Machine Learning Studio to create a model and it's pipeline and then deploy the best model and consume it. We will be using two approaches in this project to create a model:
  1. Using Azure AutoML 
  2. Using Azure HyperDrive <br/>
   
And the best model from any of the above methods will be deployed and then consumed.<br/>
We will be using `LogisticRegression classifier` to train the model and `accuracy` as a metric to check best model.

---

## Dataset

### Pima Indians Diabetes Dataset

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

https://www.kaggle.com/uciml/pima-indians-diabetes-database

#### Following are the features (columns) in the dataset:

1. `Pregnancies`: Number of times pregnant
2. `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. `BloodPressure`: Diastolic blood pressure (mm Hg)
4. `SkinThickness`: Triceps skin fold thickness (mm)
5. `Insulin`: 2-Hour serum insulin (mu U/ml)
6. `BMI`: Body mass index (weight in kg/(height in m)^2)
7. `DiabetesPedigreeFunction`: Diabetes pedigree function
8. `Age`: Age (years)
9. `Outcome`: Class variable (0 or 1)


#### Exploring the dataset

Here, I inserted the diabete dataset in Azure ml
![Dataset](https://user-images.githubusercontent.com/40363872/107894638-18f1d480-6ee5-11eb-8234-e56d206da6c6.JPG)

Here, I explored the datasets to find out about number of missing values, counts, max, min, variance and oother informations that can help us to justify our model
![Dataset explore](https://user-images.githubusercontent.com/40363872/107894708-52c2db00-6ee5-11eb-899e-ffe89af24bfc.JPG)

---

## Steps for project installation

Following steps were perform to setup the project and prepare model, pipeline, deploy and consume it:

1. Create a new `Compute Instance` or use existing one provided in the Lab, if no compute instance provided in lab create compute instance of `STANDARD_DS2_V12` type. (In our case we used Compute Instance provided by lab)
2. Create a cpu `Compute Cluster` of type `STANDARD_DS12_V2` with `4 max nodes`, and `low priority` status to train the model.
3. Register dataset to the Azure Machine Learning Workspace under `Datasets` tab and inorder to used it for creating model.
4. Uploading Notebooks and required scripts to create `AutoML` and `Hyperparameter Model`.
5. Create `AutoML` experiment through `'automl.ipynb'` notebook which performed following steps:
    - Import neccesary libraries and packages.
    - Load the dataset and compute cluster to train the model.
    - Create a new AutoML Experiment through notebook.
    - Define AutoML settings and Configuration for AutoML Run, then submit the experiment to train the model.
    - Using the `RunDetails widget`, show experiment details various models trained and accuracy achieved.

6. Create the `HyperDrive` experiment through `'hyperparameter_tuning.ipynb'` notebook as follow:
    - Import neccesary libraries and packages.
    - Load the dataset and compute cluster to train the model.
    - Create a new Experiment for HyperDrive through notebook.
    - Define `Early Termination Policy`, `Random Parameter Sampling hyperparmenter` and configuration settings.
    - Create `'train.py'` script to be used in training the model, then submit the experiment.
    - Use the `RunDetails widget` to show experiment details such as the runs accuracy rate.

8. Select the Best Model from above two approaches, Retrive the Best Model and register it in the workspace. (In our case AutoML model)
7. Then, Deploy the Best Model as a web service.
8. Enable the application insights and service logs.
9. Test the endpoint by sending a sample json payload and receive a response.


## Purpose of this Project

In this project our task is to predict wheather a user is diabetic or not based on features like number of pregnancies the patient has had, their BMI, insulin level, age, and so on and also it values.


---

## 1) Automated ML
Automated Machine Learning is the process of automating the time-consuming, iterative tasks of ML model development. It allows to build the models with high scale efficiency & productivity all while sustaining the model quality. In case of classification problem, many models such as XGBoost, RandomForest, StackEnsemble, VotingEnsemble etc. are compared

#### AutoMl configuration settings experiment
Overview of the `automl` settings and configuration settings experiment:
   - `experiment_timeout_minutes`: Set to 30 minutes. The experiment will timeout after that period to avoid over utilizing of resources.
   - `max_concurrent_iterations`: Set to 4. The max number of concurrent iterations to be run parallely.
   - `primary_metric`: Set to `'accuracy'`, which is a best suitable metrics for classification problems. 
   - `n_cross_validations`: Set to 5, therefore the training and validation sets will be divided into five equal sets.
   - `iterations`: Number of iterations for the experiment is set to 24. For number of iteration to be performed to prepare model.
   - `compute_target`: To Set project cluster used for the experiment.
   - `task`: set to `'classification'` since our target to predict whether the user is diabetic or not.
   - `training_data`: To provided the dataset which we loaded for the project.
   - `label_column_name`: Set to the result/target colunm in the dataset `'Outcome'` (0 or 1).
   - `enable_early_stopping`: Enabled to terminate the experiment if there is no improvement in model performed after few runs.
   - `featurization`: Set to `'auto'`, it's an indicator of whether implementing a featurization step to preprocess/clean the dataset automatically or not.
   - `debug_log`: For specifying a file wherein we can log everything. 

### Steps
- Following Models done after AutoML Experiment:


![AutoMl-completed](https://user-images.githubusercontent.com/40363872/107895043-b994c400-6ee6-11eb-8f9d-b7d650e05310.JPG)

This is the child run for AutoMl model

![Child run AutoMl](https://user-images.githubusercontent.com/40363872/107895076-dcbf7380-6ee6-11eb-9955-343e6704931a.JPG)

This images shows that our model pass Data guardrail tests

![Data guardrails](https://user-images.githubusercontent.com/40363872/107895086-e943cc00-6ee6-11eb-93e6-75a839ee1321.JPG)

RunDetails widget of the best model. Here, we listed our best model by AutoML.The Best Model acheived from the AutoML experiment from VotingEnsemble model. The Voting Ensemble Model which gave the accuracy of 0.79 (79%).

![RuDetails-AutoMl](https://user-images.githubusercontent.com/40363872/107895241-856dd300-6ee7-11eb-848b-14bc02670050.JPG)

Here, wI show AutoMl run with metrics in Python Jupyter

![AutoMl run with metrics](https://user-images.githubusercontent.com/40363872/107895252-96b6df80-6ee7-11eb-85f1-d88e4ecf2156.JPG)


In addition to accuracy we check other metrics including Precision, Recall and Confusion matrix for our model.

![Precision-Recall](https://user-images.githubusercontent.com/40363872/107895380-fe6d2a80-6ee7-11eb-8ada-211c275f21f8.JPG)

![Confusion matrix](https://user-images.githubusercontent.com/40363872/107895399-0c22b000-6ee8-11eb-94ee-a234f95341df.JPG)



#### Also, here we tried to find out and explore the important variables in diabete dataset and to see which independent variables has the highest influnce on predicting the outcome variable.

Here, it show important variavles by using bar-plot


![Imortant variables](https://user-images.githubusercontent.com/40363872/107895418-1fce1680-6ee8-11eb-9c5f-1642d76df100.JPG)


Also, here I show important variavles by using box-plot

![Important variables box plot](https://user-images.githubusercontent.com/40363872/107895457-40966c00-6ee8-11eb-9511-6e327138442d.JPG)



---

## 2) Hyperparameter Tuning

For HyperDrive Experiment we have chosen LogisticRegression classifier as model. 

Overview of the `Hyperparameter Tuning` settings and configuration settings experiment: 
- Parameter used for `HyperDriveConfig` settings:
  - `run_config`: In this we provide the directory wherein the `train.py` script is present and in arguments we provide `dataset id` and `compute target` details and environment by using `ScriptRunConfig` function.
  - `hyperparameter_sampling`: In this we set the parameters for sampling using `RandomParameterSampling` which includes tuning hyperparameters like for classification  `--C`: Inverse of regularization, `--max_iter`: Maximum number of iterations.
  - `policy`: In this we define the `Early termination policy` (i.e.`BanditPolicy`)  as an early stopping policy to improve the performance of the computational resources by automatically terminating poorly and delayed performing runs. Bandit Policy ends runs if the primary metric is not within the specified slack factor/amount when compared with the highest performing run.
  - `primary_metric_name`: In this we specify the metric on the basis of which we will judge the model performance. (In our case its `Accuracy`).
  - `primary_metric_goal`: In this we specify what we want the primary metric value to be (In our case `PrimaryMetricGoal.MAXIMIZE`), as we want to maximize the Accuracy.
  - `max_total_runs`: The total number of runs we want to do inorder to train the model using the above specified hyperparameters. (Set to 24)
  - `max_concurrent_runs`: The max number of concurrent iterations to be run parallely. (Set to 4)

### Steps

Details for HyperDrive Experiment with Parameter details

![Hyperdrive_Completed](https://user-images.githubusercontent.com/40363872/107895568-abe03e00-6ee8-11eb-95cb-df23e0ff71c5.JPG)


The best performing model has a 74% accuracy that is lower than AutoMl model.

![hyperdrive best model](https://user-images.githubusercontent.com/40363872/107895663-fd88c880-6ee8-11eb-8f86-cdf77225b871.JPG)


--- 

## 3. Model Deployment
Based on these two approaches, the `AutoML Experiment` gave us accuracy of **79%** while the `HyperDrive Experiment` gave accuracy of **79%**. compare to HyperDrive performance by 4.179% accuracy, So we decide to deploy AutoMl model.

- Best Model afor deployment
![AutoMl best model to deploy](https://user-images.githubusercontent.com/40363872/107895748-4ccef900-6ee9-11eb-8d7f-b8f314d90255.JPG)


- Best Model Deployment Status

![AutoMl endpoints](https://user-images.githubusercontent.com/40363872/107895774-5ce6d880-6ee9-11eb-893e-026266769319.JPG)

![AutoMl deploy 2](https://user-images.githubusercontent.com/40363872/107895847-87389600-6ee9-11eb-9a5f-aaa0ef909271.JPG)


- Consuming Endpoint and Testing
  
![Testing](https://user-images.githubusercontent.com/40363872/107895910-be0eac00-6ee9-11eb-85e0-79a733800933.JPG)

![testing 2](https://user-images.githubusercontent.com/40363872/107895939-d979b700-6ee9-11eb-81c1-c36a5457e2d0.JPG)

Also, by using Application Insights enabled, we can monitor number of failed request, Server response time and server request

![Monitoring](https://user-images.githubusercontent.com/40363872/107896092-35444000-6eea-11eb-9d65-8bacbb0f91d5.JPG)

---
## Screen Recording

Here is the link of Screen Recording [link](https://youtu.be/GKx4QSzhvu4)


---

### Future improvements:
I think by selecting the important variables that mostly effect the output variable and removing other variables we can defintly improve the accuracy of our model. 
Also, using deep learning approach such as ANN and KNN can possibly improve our model. Also we can choose the higher cross validation numbers and increase the time of iteration to improve our model accuracy. 

