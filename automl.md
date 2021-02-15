# Automated ML

Importing all needed dependencies to complete the project.


```python
import logging
import os
import json
import csv
import numpy as np
import pandas as pd
import pkg_resources
import joblib

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.core.dataset import Dataset
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.steps import AutoMLStep
from azureml.widgets import RunDetails
from azureml.core import Model, Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from sklearn.preprocessing import StandardScaler

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)
```

    SDK version: 1.20.0


## Dataset

#### Overview
This dataset comes from the Diabetes and Digestive and Kidney Disease National Institutes. 
The purpose of this dataset is to diagnose whether or not a patient is diabetic, on the basis of certain diagnostic measures in the dataset. 
The selection of these instances from a larger database was subject to several restrictions. 
All patients are women from the Indian heritage of Pima, at least 21 years old. Datasets can be found here:
https://www.kaggle.com/uciml/pima-indians-diabetes-database

#### Task
Here, we plan to predict the "Outcome" column based on the input features, either the patient has diabetes or not.


The dataset has 9 variables including:
- Pregnancies: Number pregnancy times (int).
- Glucose: Plasma glucose concentration level (int).
- BloodPressure: Diastolic blood pressure level in mm Hg(int).
- SkinThickness: skinfold thickness in mm(int).
- Insulin: two-hour serum insulin measured by mu U/ml(int).
- BMI: Body mass index (float).
- DiabetesPedigreeFunction: Diabetes pedigree function(float).
- Age: age in years 21 and above(int).
- Outcome: Target column 0 or 1, 0 = Not diabetes, 1 = diabetes(int).


```python
ws = Workspace.from_config()

# choose a name for experiment
experiment_name = 'automl-capstone'

experiment=Experiment(ws, experiment_name)
```


```python
# Load the registered dataset from workspace
dataset = Dataset.get_by_name(ws, name='diabete')

# Convert the dataset to dataframe
df = dataset.to_pandas_dataframe()
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
    Pregnancies                 768 non-null int64
    Glucose                     768 non-null int64
    BloodPressure               768 non-null int64
    SkinThickness               768 non-null int64
    Insulin                     768 non-null int64
    BMI                         768 non-null float64
    DiabetesPedigreeFunction    768 non-null float64
    Age                         768 non-null int64
    Outcome                     768 non-null int64
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB



```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.00</td>
      <td>768.00</td>
      <td>768.00</td>
      <td>768.00</td>
      <td>768.00</td>
      <td>768.00</td>
      <td>768.00</td>
      <td>768.00</td>
      <td>768.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.85</td>
      <td>120.89</td>
      <td>69.11</td>
      <td>20.54</td>
      <td>79.80</td>
      <td>31.99</td>
      <td>0.47</td>
      <td>33.24</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.37</td>
      <td>31.97</td>
      <td>19.36</td>
      <td>15.95</td>
      <td>115.24</td>
      <td>7.88</td>
      <td>0.33</td>
      <td>11.76</td>
      <td>0.48</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.08</td>
      <td>21.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.00</td>
      <td>99.00</td>
      <td>62.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>27.30</td>
      <td>0.24</td>
      <td>24.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.00</td>
      <td>117.00</td>
      <td>72.00</td>
      <td>23.00</td>
      <td>30.50</td>
      <td>32.00</td>
      <td>0.37</td>
      <td>29.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.00</td>
      <td>140.25</td>
      <td>80.00</td>
      <td>32.00</td>
      <td>127.25</td>
      <td>36.60</td>
      <td>0.63</td>
      <td>41.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.00</td>
      <td>199.00</td>
      <td>122.00</td>
      <td>99.00</td>
      <td>846.00</td>
      <td>67.10</td>
      <td>2.42</td>
      <td>81.00</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Display the first five records of the dataset
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.60</td>
      <td>0.63</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.60</td>
      <td>0.35</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.30</td>
      <td>0.67</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.10</td>
      <td>0.17</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.10</td>
      <td>2.29</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
          dtype='object')




```python
# Create CPU cluster
amlcompute_cluster_name = "notebook138912"

# Verify if cluster does not exist otherwise use the existing one
try:
    compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_DS12_V2',
                                                           vm_priority = 'lowpriority', 
                                                           max_nodes=4)
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)

```

    Found existing cluster, use it.
    
    Running


## AutoML Configuration

Overview of the automl settings and configuration used for this experiment:

- "experiment_timeout_minutes": set to 30 minutes. The experiment will timeout after that period to avoid wasting resources.
- "max_concurrent_iterations": is set to 4. The max number of concurrent iterations to be run in parallel at the same time.
- "primary_metric" : is set to 'accuracy', which is a sutible metric for classification problems.
- "n_cross_validations": is set to 5, therefore the training and validation sets will be divided into five equal sets.
- "iterations": the number of iterations for the experiment is set to 20. It's a reasonable number and would provide the intendable result for the given dataset.
- compute_target: set to the project cluster to run the experiment.
- task: set to 'classification' since our target to predict whether the patient has diabetes or not.
- training_data: the loaded dataset for the project.
- label_column_name: set to the result/target colunm in the dataset 'Outcome' (0 or 1).
- enable_early_stopping: is enabled to terminate the experiment if the accuracy score is not showing improvement over time.
- featurization = is set to 'auto', it's an indicator of whether implementing a featurization step to preprocess/clean the dataset automatically or not. In our case, the preprocessing was applied for the numerical columns which normally involve treating missing values, cluster distance, the weight of evidence...etc.
- debug_log: errors will be logged into 'automl_errors.log'.



```python
# Automl settings 
automl_settings = {
    "experiment_timeout_minutes": 30,
    "max_concurrent_iterations": 4,
    "primary_metric" : 'accuracy',
    "n_cross_validations": 5,
    "iterations": 24
    
}

# Automl config 
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = 'classification',
                             training_data=dataset,
                             label_column_name='Outcome',
                             enable_early_stopping= True,
                             featurization = 'auto',
                             debug_log = 'automl_errors.log',
                             **automl_settings
                            )


```


```python
# Submit experiment
remote_run = experiment.submit(automl_config, show_output=True)
```

    Running on remote.
    No run_configuration provided, running on notebook138912 with default configuration
    Running on remote compute: notebook138912
    Parent Run ID: AutoML_83dd20ce-82a8-4154-bfe2-a1959c1f2643
    
    Current status: FeaturesGeneration. Generating features for the dataset.
    Current status: DatasetCrossValidationSplit. Generating individually featurized CV splits.
    Current status: ModelSelection. Beginning model selection.
    
    ****************************************************************************************************
    DATA GUARDRAILS: 
    
    TYPE:         Class balancing detection
    STATUS:       PASSED
    DESCRIPTION:  Your inputs were analyzed, and all classes are balanced in your training data.
                  Learn more about imbalanced data: https://aka.ms/AutomatedMLImbalancedData
    
    ****************************************************************************************************
    
    TYPE:         Missing feature values imputation
    STATUS:       PASSED
    DESCRIPTION:  No feature missing values were detected in the training data.
                  Learn more about missing value imputation: https://aka.ms/AutomatedMLFeaturization
    
    ****************************************************************************************************
    
    TYPE:         High cardinality feature detection
    STATUS:       PASSED
    DESCRIPTION:  Your inputs were analyzed, and no high cardinality features were detected.
                  Learn more about high cardinality feature handling: https://aka.ms/AutomatedMLFeaturization
    
    ****************************************************************************************************
    
    ****************************************************************************************************
    ITERATION: The iteration being evaluated.
    PIPELINE: A summary description of the pipeline being evaluated.
    DURATION: Time taken for the current iteration.
    METRIC: The result of computing score on the fitted pipeline.
    BEST: The best observed score thus far.
    ****************************************************************************************************
    
     ITERATION   PIPELINE                                       DURATION      METRIC      BEST
             1   MaxAbsScaler XGBoostClassifier                 0:01:04       0.7656    0.7656
             0   MaxAbsScaler LightGBM                          0:01:10       0.7435    0.7656
             2   MaxAbsScaler RandomForest                      0:01:07       0.7540    0.7656
             3   MaxAbsScaler RandomForest                      0:01:17       0.7514    0.7656
             4   MaxAbsScaler RandomForest                      0:01:09       0.7578    0.7656
             5   MaxAbsScaler RandomForest                      0:01:02       0.7344    0.7656
             6   SparseNormalizer XGBoostClassifier             0:01:02       0.6993    0.7656
             7   SparseNormalizer XGBoostClassifier             0:01:00       0.7189    0.7656
            10   SparseNormalizer XGBoostClassifier             0:01:00       0.6485    0.7656
            11   StandardScalerWrapper RandomForest             0:01:00       0.7591    0.7656
             8   SparseNormalizer XGBoostClassifier             0:01:10       0.7110    0.7656
             9   SparseNormalizer LightGBM                      0:01:01       0.6759    0.7656
            12   MaxAbsScaler LogisticRegression                0:01:09       0.7683    0.7683
            13   MaxAbsScaler RandomForest                      0:01:09       0.7644    0.7683
            14   StandardScalerWrapper LogisticRegression       0:01:03       0.7617    0.7683
            15   MaxAbsScaler LightGBM                          0:01:06       0.7474    0.7683
            17   MaxAbsScaler RandomForest                      0:01:07       0.6876    0.7683
            16   SparseNormalizer XGBoostClassifier             0:01:19       0.6941    0.7683
            18   StandardScalerWrapper LightGBM                 0:01:14       0.7436    0.7683
            19   SparseNormalizer XGBoostClassifier             0:01:10       0.6928    0.7683
            20   SparseNormalizer XGBoostClassifier             0:01:04       0.6915    0.7683
            21   StandardScalerWrapper ExtremeRandomTrees       0:00:58       0.7292    0.7683
            22    VotingEnsemble                                0:01:35       0.7878    0.7878
            23    StackEnsemble                                 0:01:39       0.7734    0.7878


## Run Details

The best model has resulted from the AutoML experiment from VotingEnsemble model. The Voting Ensemble model takes a majority vote of several algorithms which makes it surpass individual algorithms and minimize the bias. 

Use the `RunDetails` widget to show the different experiments.


```python
RunDetails(remote_run).show()
```


    _AutoMLWidget(widget_settings={'childWidgetDisplay': 'popup', 'send_telemetry': False, 'log_level': 'INFO', 's…





```python
remote_run.wait_for_completion(show_output=True)
```

    
    
    ****************************************************************************************************
    DATA GUARDRAILS: 
    
    TYPE:         Class balancing detection
    STATUS:       PASSED
    DESCRIPTION:  Your inputs were analyzed, and all classes are balanced in your training data.
                  Learn more about imbalanced data: https://aka.ms/AutomatedMLImbalancedData
    
    ****************************************************************************************************
    
    TYPE:         Missing feature values imputation
    STATUS:       PASSED
    DESCRIPTION:  No feature missing values were detected in the training data.
                  Learn more about missing value imputation: https://aka.ms/AutomatedMLFeaturization
    
    ****************************************************************************************************
    
    TYPE:         High cardinality feature detection
    STATUS:       PASSED
    DESCRIPTION:  Your inputs were analyzed, and no high cardinality features were detected.
                  Learn more about high cardinality feature handling: https://aka.ms/AutomatedMLFeaturization
    
    ****************************************************************************************************
    
    ****************************************************************************************************
    ITERATION: The iteration being evaluated.
    PIPELINE: A summary description of the pipeline being evaluated.
    DURATION: Time taken for the current iteration.
    METRIC: The result of computing score on the fitted pipeline.
    BEST: The best observed score thus far.
    ****************************************************************************************************
    
     ITERATION   PIPELINE                                       DURATION      METRIC      BEST
             0   MaxAbsScaler LightGBM                          0:01:10       0.7435    0.7435
             1   MaxAbsScaler XGBoostClassifier                 0:01:04       0.7656    0.7656
             2   MaxAbsScaler RandomForest                      0:01:07       0.7540    0.7656
             3   MaxAbsScaler RandomForest                      0:01:17       0.7514    0.7656
             4   MaxAbsScaler RandomForest                      0:01:09       0.7578    0.7656
             5   MaxAbsScaler RandomForest                      0:01:02       0.7344    0.7656
             6   SparseNormalizer XGBoostClassifier             0:01:02       0.6993    0.7656
             7   SparseNormalizer XGBoostClassifier             0:01:00       0.7189    0.7656
            10   SparseNormalizer XGBoostClassifier             0:01:00       0.6485    0.7656
            11   StandardScalerWrapper RandomForest             0:01:00       0.7591    0.7656
             8   SparseNormalizer XGBoostClassifier             0:01:10       0.7110    0.7656
             9   SparseNormalizer LightGBM                      0:01:01       0.6759    0.7656
            12   MaxAbsScaler LogisticRegression                0:01:09       0.7683    0.7683
            13   MaxAbsScaler RandomForest                      0:01:09       0.7644    0.7683
            14   StandardScalerWrapper LogisticRegression       0:01:03       0.7617    0.7683
            15   MaxAbsScaler LightGBM                          0:01:06       0.7474    0.7683
            16   SparseNormalizer XGBoostClassifier             0:01:19       0.6941    0.7683
            17   MaxAbsScaler RandomForest                      0:01:07       0.6876    0.7683
            18   StandardScalerWrapper LightGBM                 0:01:14       0.7436    0.7683
            19   SparseNormalizer XGBoostClassifier             0:01:10       0.6928    0.7683
            20   SparseNormalizer XGBoostClassifier             0:01:04       0.6915    0.7683
            21   StandardScalerWrapper ExtremeRandomTrees       0:00:58       0.7292    0.7683
            22    VotingEnsemble                                0:01:35       0.7878    0.7878
            23    StackEnsemble                                 0:01:39       0.7734    0.7878





    {'runId': 'AutoML_83dd20ce-82a8-4154-bfe2-a1959c1f2643',
     'target': 'notebook138912',
     'status': 'Completed',
     'startTimeUtc': '2021-02-14T23:37:37.805919Z',
     'endTimeUtc': '2021-02-14T23:51:14.343343Z',
     'properties': {'num_iterations': '24',
      'training_type': 'TrainFull',
      'acquisition_function': 'EI',
      'primary_metric': 'accuracy',
      'train_split': '0',
      'acquisition_parameter': '0',
      'num_cross_validation': '5',
      'target': 'notebook138912',
      'AMLSettingsJsonString': '{"path":null,"name":"automl-capstone","subscription_id":"b968fb36-f06a-4c76-a15f-afab68ae7667","resource_group":"aml-quickstarts-138912","workspace_name":"quick-starts-ws-138912","region":"southcentralus","compute_target":"notebook138912","spark_service":null,"azure_service":"remote","many_models":false,"pipeline_fetch_max_batch_size":1,"iterations":24,"primary_metric":"accuracy","task_type":"classification","data_script":null,"validation_size":0.0,"n_cross_validations":5,"y_min":null,"y_max":null,"num_classes":null,"featurization":"auto","_ignore_package_version_incompatibilities":false,"is_timeseries":false,"max_cores_per_iteration":1,"max_concurrent_iterations":4,"iteration_timeout_minutes":null,"mem_in_mb":null,"enforce_time_on_windows":false,"experiment_timeout_minutes":30,"experiment_exit_score":null,"whitelist_models":null,"blacklist_algos":["TensorFlowLinearClassifier","TensorFlowDNN"],"supported_models":["KNN","LightGBM","ExtremeRandomTrees","XGBoostClassifier","AveragedPerceptronClassifier","SGD","LinearSVM","TensorFlowLinearClassifier","BernoulliNaiveBayes","MultinomialNaiveBayes","GradientBoosting","TensorFlowDNN","DecisionTree","SVM","LogisticRegression","RandomForest"],"auto_blacklist":true,"blacklist_samples_reached":false,"exclude_nan_labels":true,"verbosity":20,"_debug_log":"azureml_automl.log","show_warnings":false,"model_explainability":true,"service_url":null,"sdk_url":null,"sdk_packages":null,"enable_onnx_compatible_models":false,"enable_split_onnx_featurizer_estimator_models":false,"vm_type":"STANDARD_DS3_V2","telemetry_verbosity":20,"send_telemetry":true,"enable_dnn":false,"scenario":"SDK-1.13.0","environment_label":null,"force_text_dnn":false,"enable_feature_sweeping":true,"enable_early_stopping":true,"early_stopping_n_iters":10,"metrics":null,"enable_ensembling":true,"enable_stack_ensembling":true,"ensemble_iterations":15,"enable_tf":false,"enable_subsampling":false,"subsample_seed":null,"enable_nimbusml":false,"enable_streaming":false,"force_streaming":false,"track_child_runs":true,"allowed_private_models":[],"label_column_name":"Outcome","weight_column_name":null,"cv_split_column_names":null,"enable_local_managed":false,"_local_managed_run_id":null,"cost_mode":1,"lag_length":0,"metric_operation":"maximize","preprocess":true}',
      'DataPrepJsonString': '{\\"training_data\\": \\"{\\\\\\"blocks\\\\\\": [{\\\\\\"id\\\\\\": \\\\\\"ad8ce605-e401-41ed-a697-c5789cdc30e0\\\\\\", \\\\\\"type\\\\\\": \\\\\\"Microsoft.DPrep.GetDatastoreFilesBlock\\\\\\", \\\\\\"arguments\\\\\\": {\\\\\\"datastores\\\\\\": [{\\\\\\"datastoreName\\\\\\": \\\\\\"workspaceblobstore\\\\\\", \\\\\\"path\\\\\\": \\\\\\"UI/02-14-2021_091821_UTC/diabetes.csv\\\\\\", \\\\\\"resourceGroup\\\\\\": \\\\\\"aml-quickstarts-138912\\\\\\", \\\\\\"subscription\\\\\\": \\\\\\"b968fb36-f06a-4c76-a15f-afab68ae7667\\\\\\", \\\\\\"workspaceName\\\\\\": \\\\\\"quick-starts-ws-138912\\\\\\"}]}, \\\\\\"localData\\\\\\": {}, \\\\\\"isEnabled\\\\\\": true, \\\\\\"name\\\\\\": null, \\\\\\"annotation\\\\\\": null}, {\\\\\\"id\\\\\\": \\\\\\"8b2a8ab9-9924-46bf-9517-932ca18ce821\\\\\\", \\\\\\"type\\\\\\": \\\\\\"Microsoft.DPrep.ParseDelimitedBlock\\\\\\", \\\\\\"arguments\\\\\\": {\\\\\\"columnHeadersMode\\\\\\": 3, \\\\\\"fileEncoding\\\\\\": 0, \\\\\\"handleQuotedLineBreaks\\\\\\": false, \\\\\\"preview\\\\\\": false, \\\\\\"separator\\\\\\": \\\\\\",\\\\\\", \\\\\\"skipRows\\\\\\": 0, \\\\\\"skipRowsMode\\\\\\": 0}, \\\\\\"localData\\\\\\": {}, \\\\\\"isEnabled\\\\\\": true, \\\\\\"name\\\\\\": null, \\\\\\"annotation\\\\\\": null}, {\\\\\\"id\\\\\\": \\\\\\"14ca6b58-0031-48dd-9659-bb14bc0365bd\\\\\\", \\\\\\"type\\\\\\": \\\\\\"Microsoft.DPrep.DropColumnsBlock\\\\\\", \\\\\\"arguments\\\\\\": {\\\\\\"columns\\\\\\": {\\\\\\"type\\\\\\": 0, \\\\\\"details\\\\\\": {\\\\\\"selectedColumns\\\\\\": [\\\\\\"Path\\\\\\"]}}}, \\\\\\"localData\\\\\\": {}, \\\\\\"isEnabled\\\\\\": true, \\\\\\"name\\\\\\": null, \\\\\\"annotation\\\\\\": null}, {\\\\\\"id\\\\\\": \\\\\\"72b34afb-1959-4848-b25d-8f4941c32b11\\\\\\", \\\\\\"type\\\\\\": \\\\\\"Microsoft.DPrep.SetColumnTypesBlock\\\\\\", \\\\\\"arguments\\\\\\": {\\\\\\"columnConversion\\\\\\": [{\\\\\\"column\\\\\\": {\\\\\\"type\\\\\\": 2, \\\\\\"details\\\\\\": {\\\\\\"selectedColumn\\\\\\": \\\\\\"Path\\\\\\"}}, \\\\\\"typeProperty\\\\\\": 0}, {\\\\\\"column\\\\\\": {\\\\\\"type\\\\\\": 2, \\\\\\"details\\\\\\": {\\\\\\"selectedColumn\\\\\\": \\\\\\"Pregnancies\\\\\\"}}, \\\\\\"typeProperty\\\\\\": 2}, {\\\\\\"column\\\\\\": {\\\\\\"type\\\\\\": 2, \\\\\\"details\\\\\\": {\\\\\\"selectedColumn\\\\\\": \\\\\\"Glucose\\\\\\"}}, \\\\\\"typeProperty\\\\\\": 2}, {\\\\\\"column\\\\\\": {\\\\\\"type\\\\\\": 2, \\\\\\"details\\\\\\": {\\\\\\"selectedColumn\\\\\\": \\\\\\"BloodPressure\\\\\\"}}, \\\\\\"typeProperty\\\\\\": 2}, {\\\\\\"column\\\\\\": {\\\\\\"type\\\\\\": 2, \\\\\\"details\\\\\\": {\\\\\\"selectedColumn\\\\\\": \\\\\\"SkinThickness\\\\\\"}}, \\\\\\"typeProperty\\\\\\": 2}, {\\\\\\"column\\\\\\": {\\\\\\"type\\\\\\": 2, \\\\\\"details\\\\\\": {\\\\\\"selectedColumn\\\\\\": \\\\\\"Insulin\\\\\\"}}, \\\\\\"typeProperty\\\\\\": 2}, {\\\\\\"column\\\\\\": {\\\\\\"type\\\\\\": 2, \\\\\\"details\\\\\\": {\\\\\\"selectedColumn\\\\\\": \\\\\\"BMI\\\\\\"}}, \\\\\\"typeProperty\\\\\\": 3}, {\\\\\\"column\\\\\\": {\\\\\\"type\\\\\\": 2, \\\\\\"details\\\\\\": {\\\\\\"selectedColumn\\\\\\": \\\\\\"DiabetesPedigreeFunction\\\\\\"}}, \\\\\\"typeProperty\\\\\\": 3}, {\\\\\\"column\\\\\\": {\\\\\\"type\\\\\\": 2, \\\\\\"details\\\\\\": {\\\\\\"selectedColumn\\\\\\": \\\\\\"Age\\\\\\"}}, \\\\\\"typeProperty\\\\\\": 2}, {\\\\\\"column\\\\\\": {\\\\\\"type\\\\\\": 2, \\\\\\"details\\\\\\": {\\\\\\"selectedColumn\\\\\\": \\\\\\"Outcome\\\\\\"}}, \\\\\\"typeProperty\\\\\\": 2}]}, \\\\\\"localData\\\\\\": {}, \\\\\\"isEnabled\\\\\\": true, \\\\\\"name\\\\\\": null, \\\\\\"annotation\\\\\\": null}], \\\\\\"inspectors\\\\\\": [], \\\\\\"meta\\\\\\": {\\\\\\"savedDatasetId\\\\\\": \\\\\\"f18913f8-b280-4eb9-94b1-b18249536a79\\\\\\", \\\\\\"datasetType\\\\\\": \\\\\\"tabular\\\\\\", \\\\\\"subscriptionId\\\\\\": \\\\\\"b968fb36-f06a-4c76-a15f-afab68ae7667\\\\\\", \\\\\\"workspaceId\\\\\\": \\\\\\"8a3e3d52-5ff2-4ce7-8f68-93f8c1aeaac6\\\\\\", \\\\\\"workspaceLocation\\\\\\": \\\\\\"southcentralus\\\\\\"}}\\", \\"activities\\": 0}',
      'EnableSubsampling': 'False',
      'runTemplate': 'AutoML',
      'azureml.runsource': 'automl',
      'display_task_type': 'classification',
      'dependencies_versions': '{"azureml-widgets": "1.20.0", "azureml-train": "1.20.0", "azureml-train-restclients-hyperdrive": "1.20.0", "azureml-train-core": "1.20.0", "azureml-train-automl": "1.20.0", "azureml-train-automl-runtime": "1.20.0", "azureml-train-automl-client": "1.20.0", "azureml-tensorboard": "1.20.0", "azureml-telemetry": "1.20.0", "azureml-sdk": "1.20.0", "azureml-samples": "0+unknown", "azureml-pipeline": "1.20.0", "azureml-pipeline-steps": "1.20.0", "azureml-pipeline-core": "1.20.0", "azureml-opendatasets": "1.20.0", "azureml-model-management-sdk": "1.0.1b6.post1", "azureml-mlflow": "1.20.0.post1", "azureml-interpret": "1.20.0", "azureml-explain-model": "1.20.0", "azureml-defaults": "1.20.0", "azureml-dataset-runtime": "1.20.0", "azureml-dataprep": "2.7.3", "azureml-dataprep-rslex": "1.5.0", "azureml-dataprep-native": "27.0.0", "azureml-datadrift": "1.20.0", "azureml-core": "1.20.0", "azureml-contrib-services": "1.20.0", "azureml-contrib-server": "1.20.0", "azureml-contrib-reinforcementlearning": "1.20.0", "azureml-contrib-pipeline-steps": "1.20.0", "azureml-contrib-notebook": "1.20.0", "azureml-contrib-interpret": "1.20.0", "azureml-contrib-gbdt": "1.20.0", "azureml-contrib-fairness": "1.20.0", "azureml-contrib-dataset": "1.20.0", "azureml-cli-common": "1.20.0", "azureml-automl-runtime": "1.20.0", "azureml-automl-core": "1.20.0", "azureml-accel-models": "1.20.0"}',
      '_aml_system_scenario_identification': 'Remote.Parent',
      'ClientType': 'SDK',
      'environment_cpu_name': 'AzureML-AutoML',
      'environment_cpu_label': 'prod',
      'environment_gpu_name': 'AzureML-AutoML-GPU',
      'environment_gpu_label': 'prod',
      'root_attribution': 'automl',
      'attribution': 'AutoML',
      'Orchestrator': 'AutoML',
      'CancelUri': 'https://southcentralus.experiments.azureml.net/jasmine/v1.0/subscriptions/b968fb36-f06a-4c76-a15f-afab68ae7667/resourceGroups/aml-quickstarts-138912/providers/Microsoft.MachineLearningServices/workspaces/quick-starts-ws-138912/experimentids/dae083dc-6d3b-4aa8-8ac1-b99ec064a02d/cancel/AutoML_83dd20ce-82a8-4154-bfe2-a1959c1f2643',
      'ClientSdkVersion': '1.21.0',
      'snapshotId': '00000000-0000-0000-0000-000000000000',
      'SetupRunId': 'AutoML_83dd20ce-82a8-4154-bfe2-a1959c1f2643_setup',
      'SetupRunContainerId': 'dcid.AutoML_83dd20ce-82a8-4154-bfe2-a1959c1f2643_setup',
      'FeaturizationRunJsonPath': 'featurizer_container.json',
      'FeaturizationRunId': 'AutoML_83dd20ce-82a8-4154-bfe2-a1959c1f2643_featurize',
      'ProblemInfoJsonString': '{"dataset_num_categorical": 0, "is_sparse": true, "subsampling": false, "dataset_classes": 2, "dataset_features": 24, "dataset_samples": 768, "single_frequency_class_detected": false}',
      'ModelExplainRunId': 'AutoML_83dd20ce-82a8-4154-bfe2-a1959c1f2643_ModelExplain'},
     'inputDatasets': [{'dataset': {'id': 'f18913f8-b280-4eb9-94b1-b18249536a79'}, 'consumptionDetails': {'type': 'RunInput', 'inputName': 'training_data', 'mechanism': 'Direct'}}],
     'outputDatasets': [],
     'logFiles': {},
     'submittedBy': 'ODL_User 138912'}




```python
remote_run
```




<table style="width:100%"><tr><th>Experiment</th><th>Id</th><th>Type</th><th>Status</th><th>Details Page</th><th>Docs Page</th></tr><tr><td>automl-capstone</td><td>AutoML_83dd20ce-82a8-4154-bfe2-a1959c1f2643</td><td>automl</td><td>Completed</td><td><a href="https://ml.azure.com/experiments/automl-capstone/runs/AutoML_83dd20ce-82a8-4154-bfe2-a1959c1f2643?wsid=/subscriptions/b968fb36-f06a-4c76-a15f-afab68ae7667/resourcegroups/aml-quickstarts-138912/workspaces/quick-starts-ws-138912" target="_blank" rel="noopener">Link to Azure Machine Learning studio</a></td><td><a href="https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py" target="_blank" rel="noopener">Link to Documentation</a></td></tr></table>



## Best Model

The cell below shows the best model from the automl experiments and display all the properties of the model.




```python
# find and save best automl model
best_run, fitted_model = remote_run.get_output()
print(best_run)
print(fitted_model.steps)
```

    WARNING:root:The version of the SDK does not match the version the model was trained on.
    WARNING:root:The consistency in the result may not be guaranteed.
    WARNING:root:Package:azureml-automl-core, training version:1.21.0, current version:1.20.0
    Package:azureml-automl-runtime, training version:1.21.0, current version:1.20.0
    Package:azureml-core, training version:1.21.0.post1, current version:1.20.0
    Package:azureml-dataprep, training version:2.8.2, current version:2.7.3
    Package:azureml-dataprep-native, training version:28.0.0, current version:27.0.0
    Package:azureml-dataprep-rslex, training version:1.6.0, current version:1.5.0
    Package:azureml-dataset-runtime, training version:1.21.0, current version:1.20.0
    Package:azureml-defaults, training version:1.21.0, current version:1.20.0
    Package:azureml-interpret, training version:1.21.0, current version:1.20.0
    Package:azureml-pipeline-core, training version:1.21.0, current version:1.20.0
    Package:azureml-telemetry, training version:1.21.0, current version:1.20.0
    Package:azureml-train-automl-client, training version:1.21.0, current version:1.20.0
    Package:azureml-train-automl-runtime, training version:1.21.0, current version:1.20.0
    WARNING:root:Please ensure the version of your local conda dependencies match the version on which your model was trained in order to properly retrieve your model.


    Run(Experiment: automl-capstone,
    Id: AutoML_83dd20ce-82a8-4154-bfe2-a1959c1f2643_22,
    Type: azureml.scriptrun,
    Status: Completed)
    [('datatransformer', DataTransformer(enable_dnn=None, enable_feature_sweeping=None,
                    feature_sweeping_config=None, feature_sweeping_timeout=None,
                    featurization_config=None, force_text_dnn=None,
                    is_cross_validation=None, is_onnx_compatible=None, logger=None,
                    observer=None, task=None, working_dir=None)), ('prefittedsoftvotingclassifier', PreFittedSoftVotingClassifier(classification_labels=None,
                                  estimators=[('12',
                                               Pipeline(memory=None,
                                                        steps=[('maxabsscaler',
                                                                MaxAbsScaler(copy=True)),
                                                               ('logisticregression',
                                                                LogisticRegression(C=2.559547922699533,
                                                                                   class_weight=None,
                                                                                   dual=False,
                                                                                   fit_intercept=True,
                                                                                   intercept_scaling=1,
                                                                                   l1_ratio=None,
                                                                                   max_iter=100,
                                                                                   multi_class='ovr',
                                                                                   n_jobs=1,
                                                                                   penalty='l2',
                                                                                   random_sta...
                                                                                  reg_lambda=0.20833333333333334,
                                                                                  scale_pos_weight=1,
                                                                                  seed=None,
                                                                                  silent=None,
                                                                                  subsample=1,
                                                                                  tree_method='auto',
                                                                                  verbose=-10,
                                                                                  verbosity=0))],
                                                        verbose=False))],
                                  flatten_transform=None,
                                  weights=[0.07692307692307693, 0.15384615384615385,
                                           0.07692307692307693, 0.07692307692307693,
                                           0.23076923076923078, 0.07692307692307693,
                                           0.15384615384615385, 0.07692307692307693,
                                           0.07692307692307693]))]



```python
print(fitted_model)
```

    Pipeline(memory=None,
             steps=[('datatransformer',
                     DataTransformer(enable_dnn=None, enable_feature_sweeping=None,
                                     feature_sweeping_config=None,
                                     feature_sweeping_timeout=None,
                                     featurization_config=None, force_text_dnn=None,
                                     is_cross_validation=None,
                                     is_onnx_compatible=None, logger=None,
                                     observer=None, task=None, working_dir=None)),
                    ('prefittedsoftvotingclassifier',...
                                                                                                   reg_alpha=0.5208333333333334,
                                                                                                   reg_lambda=0.8333333333333334,
                                                                                                   scale_pos_weight=1,
                                                                                                   seed=None,
                                                                                                   silent=None,
                                                                                                   subsample=0.6,
                                                                                                   tree_method='auto',
                                                                                                   verbose=-10,
                                                                                                   verbosity=0))],
                                                                         verbose=False))],
                                                   flatten_transform=None,
                                                   weights=[0.09090909090909091,
                                                            0.09090909090909091,
                                                            0.09090909090909091,
                                                            0.18181818181818182,
                                                            0.18181818181818182,
                                                            0.18181818181818182,
                                                            0.18181818181818182]))],
             verbose=False)



```python
best_run_metrics=remote_run.get_metrics()
```


```python
# Show all metrics for our best run
for metric_name in best_run_metrics:
    metric = best_run_metrics[metric_name]
    print(metric_name, metric)
```

    experiment_status ['DatasetEvaluation', 'FeaturesGeneration', 'DatasetFeaturization', 'DatasetFeaturizationCompleted', 'DatasetCrossValidationSplit', 'ModelSelection', 'BestRunExplainModel', 'ModelExplanationDataSetSetup', 'PickSurrogateModel', 'EngineeredFeatureExplanations', 'EngineeredFeatureExplanations', 'RawFeaturesExplanations', 'RawFeaturesExplanations', 'BestRunExplainModel']
    experiment_status_description ['Gathering dataset statistics.', 'Generating features for the dataset.', 'Beginning to fit featurizers and featurize the dataset.', 'Completed fit featurizers and featurizing the dataset.', 'Generating individually featurized CV splits.', 'Beginning model selection.', 'Best run model explanations started', 'Model explanations data setup completed', 'Choosing LightGBM as the surrogate model for explanations', 'Computation of engineered features started', 'Computation of engineered features completed', 'Computation of raw features started', 'Computation of raw features completed', 'Best run model explanations completed']
    log_loss 0.4885603356226982
    matthews_correlation 0.5098564336296645
    recall_score_micro 0.787810881928529
    average_precision_score_macro 0.8183976441002858
    precision_score_weighted 0.7880618303762439
    weighted_accuracy 0.8319315074016357
    f1_score_macro 0.7436454383099261
    AUC_macro 0.8384187551192497
    balanced_accuracy 0.7313159384154433
    recall_score_macro 0.7313159384154433
    AUC_micro 0.854130468694797
    average_precision_score_micro 0.8558965824636541
    precision_score_micro 0.787810881928529
    norm_macro_recall 0.46263187683088675
    f1_score_micro 0.787810881928529
    precision_score_macro 0.7810460266904766
    AUC_weighted 0.8384187551192497
    recall_score_weighted 0.787810881928529
    average_precision_score_weighted 0.8477435537592358
    accuracy 0.787810881928529
    f1_score_weighted 0.7774961223704373



```python
BestModel = best_run.register_model(model_path='outputs/model.pkl', model_name='diabeteModel_automl',
                        tags={'Training context':'Auto ML'},
                        properties={'Accuracy': best_run_metrics['accuracy']})

print(BestModel)
```

    Model(workspace=Workspace.create(name='quick-starts-ws-138912', subscription_id='b968fb36-f06a-4c76-a15f-afab68ae7667', resource_group='aml-quickstarts-138912'), name=diabeteModel_automl, id=diabeteModel_automl:3, version=3, tags={'Training context': 'Auto ML'}, properties={'Accuracy': '0.787810881928529'})



```python
# SHow that our model saved
for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')
```

    diabeteModel_automl version: 3
    	 Training context : Auto ML
    	 Accuracy : 0.787810881928529
    
    
    automl-best-model version: 1
    
    
    diabeteModel_automl version: 2
    
    
    diabeteModel_automl version: 1
    	 Training context : Auto ML
    	 Accuracy : 0.787810881928529
    
    
    capstoneModel_automl version: 1
    	 Training context : Auto ML
    	 Accuracy : 0.787810881928529
    
    


## Model Deployment

Remember you have to deploy only one of the two models you trained.. Perform the steps in the rest of this notebook only if you wish to deploy this model.

In the cell below, we have registered the model, created an inference config and deployed the model as a web service.


```python
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice.aci import AciWebservice
from azureml.core.webservice import Webservice
from azureml.core.model import Model
```


```python
script_file= 'score.py'

best_run.download_file('outputs/scoring_file_v_1_0_0.py', script_file)
```


```python
# Registring the best model
model = best_model.register_model(model_name='automl-best-model',model_path='outputs/model.pkl')

```


```python
inference_config = InferenceConfig(entry_script='score.py',
 environment=environment)

```


```python
service_name = 'automl-deploy'
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config,
                       overwrite=True
                      )
service.wait_for_deployment(show_output=True)

scoring_uri = service.scoring_uri
print(scoring_uri)
```

    Tips: You can try get_logs(): https://aka.ms/debugimage#dockerlog or local deployment: https://aka.ms/debugimage#debug-locally to debug if deployment takes longer than 10 minutes.
    Running.......................................................
    Succeeded
    ACI service creation operation finished, operation "Succeeded"
    http://f7511453-d747-4906-b13b-8de05ba86490.southcentralus.azurecontainer.io/score



```python
# Enable app insights
service.update(enable_app_insights=True)
```

In the cell below, we have sent a request to the web service to test it.


```python
import requests
import json
```


```python


#two set of data to score, so we get two results back
data = {"data": [{"Pregnancies": 5, 
     "Glucose": 150, 
     "BloodPressure": 70, 
     "SkinThickness": 40, 
     "Insulin": 10, 
     "BMI": 36.5, 
     "DiabetesPedigreeFunction": 0.627, 
     "Age": 30},

    {"Pregnancies": 5, 
     "Glucose": 90, 
     "BloodPressure": 70, 
     "SkinThickness": 34, 
     "Insulin": 20, 
     "BMI": 26.5, 
     "DiabetesPedigreeFunction": 0.351, 
     "Age": 28},
      ]}
    
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
#headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
print("Case 0: Not Diabetes, Case 1: Diabetes.")
```

    {"result": [1, 0]}
    Case 0: Not Diabetes, Case 1: Diabetes.



```python
data = [{"Pregnancies": 12, 
     "Glucose": 190, 
     "BloodPressure": 120, 
     "SkinThickness": 60, 
     "Insulin": 700, 
     "BMI": 56.5, 
     "DiabetesPedigreeFunction": 1.5, 
     "Age": 70},

    {"Pregnancies": 6, 
     "Glucose": 40, 
     "BloodPressure": 70, 
     "SkinThickness": 24, 
     "Insulin": 0, 
     "BMI": 25.5, 
     "DiabetesPedigreeFunction": 0.351, 
     "Age": 22},
      ]

print(data)
```

    [{'Pregnancies': 12, 'Glucose': 190, 'BloodPressure': 120, 'SkinThickness': 60, 'Insulin': 700, 'BMI': 56.5, 'DiabetesPedigreeFunction': 1.5, 'Age': 70}, {'Pregnancies': 6, 'Glucose': 40, 'BloodPressure': 70, 'SkinThickness': 24, 'Insulin': 0, 'BMI': 25.5, 'DiabetesPedigreeFunction': 0.351, 'Age': 22}]



```python
# test using service instance
input_data = json.dumps({
    'data': data
})

output = service.run(input_data)
output
```




    '{"result": [1, 0]}'



Print the logs of the web service and delete the service


```python
logs = service.get_logs()
logs
```




    '2021-02-15T00:04:53,498133000+00:00 - gunicorn/run \n2021-02-15T00:04:53,526219700+00:00 - iot-server/run \n2021-02-15T00:04:53,545821200+00:00 - rsyslog/run \n2021-02-15T00:04:53,556377000+00:00 - nginx/run \n/usr/sbin/nginx: /azureml-envs/azureml_09ff55f546b313bb1ab136a466214499/lib/libcrypto.so.1.0.0: no version information available (required by /usr/sbin/nginx)\n/usr/sbin/nginx: /azureml-envs/azureml_09ff55f546b313bb1ab136a466214499/lib/libcrypto.so.1.0.0: no version information available (required by /usr/sbin/nginx)\n/usr/sbin/nginx: /azureml-envs/azureml_09ff55f546b313bb1ab136a466214499/lib/libssl.so.1.0.0: no version information available (required by /usr/sbin/nginx)\n/usr/sbin/nginx: /azureml-envs/azureml_09ff55f546b313bb1ab136a466214499/lib/libssl.so.1.0.0: no version information available (required by /usr/sbin/nginx)\n/usr/sbin/nginx: /azureml-envs/azureml_09ff55f546b313bb1ab136a466214499/lib/libssl.so.1.0.0: no version information available (required by /usr/sbin/nginx)\nrsyslogd: /azureml-envs/azureml_09ff55f546b313bb1ab136a466214499/lib/libuuid.so.1: no version information available (required by rsyslogd)\nEdgeHubConnectionString and IOTEDGE_IOTHUBHOSTNAME are not set. Exiting...\n2021-02-15T00:04:53,857488000+00:00 - iot-server/finish 1 0\n2021-02-15T00:04:53,885506500+00:00 - Exit code 1 is normal. Not restarting iot-server.\nStarting gunicorn 19.9.0\nListening at: http://127.0.0.1:31311 (123)\nUsing worker: sync\nworker timeout is set to 300\nBooting worker with pid: 151\nSPARK_HOME not set. Skipping PySpark Initialization.\nGenerating new fontManager, this may take some time...\nInitializing logger\n2021-02-15 00:05:00,069 | root | INFO | Starting up app insights client\n2021-02-15 00:05:00,071 | root | INFO | Starting up request id generator\n2021-02-15 00:05:00,072 | root | INFO | Starting up app insight hooks\n2021-02-15 00:05:00,072 | root | INFO | Invoking user\'s init function\n2021-02-15 00:05:11,391 | root | INFO | Users\'s init has completed successfully\n2021-02-15 00:05:11,403 | root | INFO | Skipping middleware: dbg_model_info as it\'s not enabled.\n2021-02-15 00:05:11,404 | root | INFO | Skipping middleware: dbg_resource_usage as it\'s not enabled.\n2021-02-15 00:05:11,410 | root | INFO | Scoring timeout is found from os.environ: 60000 ms\n2021-02-15 00:05:14,372 | root | INFO | 200\n127.0.0.1 - - [15/Feb/2021:00:05:14 +0000] "GET /swagger.json HTTP/1.0" 200 2536 "-" "Go-http-client/1.1"\n2021-02-15 00:05:26,193 | root | INFO | Validation Request Content-Type\n2021-02-15 00:05:26,194 | root | INFO | Scoring Timer is set to 60.0 seconds\n2021-02-15 00:05:26,304 | root | INFO | 200\n127.0.0.1 - - [15/Feb/2021:00:05:26 +0000] "POST /score HTTP/1.0" 200 22 "-" "python-requests/2.25.1"\n'




```python
service.delete()
```


```python

```


```python

```
