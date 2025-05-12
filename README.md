
# Rodent Infestation Prediction

## Value proposition

### Target: NYC Department of Health and Mental Hygiene
- Existing business model: The New York City Department of Health and Mental Hygiene (DOHMH) uses a multi-pronged approach to address rodent infestations, combining public health interventions with private sector involvement. DOHMH primarily focuses on preventing and controlling infestations through inspections.
- Our value proposition: Use historic inspection data along with weather, rat complaints and building permits to preemptively predict the probability of infestation occurrence in restaurants belonging to a pre-defined geographic radius 
- Hepls health inspectors identify restaurants with the maximum likelyhood of rodent infestations and allows them to target restaurantss

## Scale

### 1) Offline data: 11GB
- Rodent complaints (2GB), Weather data (1GB), building permits (7GB), restaurant inspection data (1GB)

### 2) Models: 10 models
- 5 XGBoost models for each borough 
- 5 GAT models for each borough

### 3) Throughtput and training time
- The model takes 10 seconds for each inference
- It takes 2 hours to train all models on 2 node mi100 GPUs
---


## Contributors


| Name                            | Responsible for                      | Link to their commits in this repo |
|---------------------------------|--------------------------------------|------------------------------------|
| Aditya Krishna                  |Model serving and monitoring platforms|[Aditya's commits](https://github.com/adkrish1/Rodent-Infestation-Prediction/commits/main/?author=adkrish1)|
| Akshay Hemant Paralikar         |Model training and training platforms |[Akshay's commits](https://github.com/adkrish1/Rodent-Infestation-Prediction/commits/main/?author=akshay412)|
| Kunal Thadani                   |Continious X                          |[Kunal's commits](https://github.com/adkrish1/Rodent-Infestation-Prediction/commits/main/?author=kunalthadani)|
| Rakshith Murthy                 |Data Pipeline                         |[Rakshith's commits](https://github.com/adkrish1/Rodent-Infestation-Prediction/commits/main/?author=valar007)|

## Cloud-native
![alt text](diagram.jpg)

### Infrastructure and Infrastructure-as-code: : 
Terraform and ansible are used to provision resources and attach block storage. Ansible playbooks are used to provision Kubernetes, ArgoCD and Argo Workflows using kubespray. We can bring up VMs in KVM@TACC using terraform.

- Terraform: https://github.com/adkrish1/Rodent-Infestation-Prediction/tree/main/Continous_X/tf
- Ansible: https://github.com/adkrish1/Rodent-Infestation-Prediction/tree/main/Continous_X/ansible

## Data Pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

- Persistant storage: We make use of 30GB of both block and volume storage. Block storage consists of all the application's data, models, RayTrain and the container data. Object store consists of downloaded datasets, transformed datasets. [object store](https://chi.tacc.chameleoncloud.org/project/containers/container/object-persist-project8) [volume store](https://kvm.tacc.chameleoncloud.org/project/volumes/63a05616-57eb-4ab6-9342-a0184ee9f12e/)
- Offline data: This data input to the model is obtained after transformations. Example data:

| boro     | camis    | dba                   | latitude        | longitude        | month   | rat_complaints_0.1mi | rat_complaints_1.0mi | building_count_0.5mi |
|----------|----------|-----------------------|-----------------|------------------|---------|----------------------|----------------------|----------------------|
| Bronx    | 30075445 | MORRIS PARK BAKE SHOP | 40.848231224526 | -73.855971889932 | 2023-01 | 0                    | 21                   | 32                   |
| Brooklyn | 50168612 | 3824 MUNCHIES LLC     | 40.590564056548 | -73.940000831706 | 2025-01 | 0                    | 11                   | 65                   |

- The production data is not known until the health inspector visits the restaurant and inspects the establishment
- We used 4 datasets: restaurant inspection data (camis, dba, lat, long, score, violation_code, inspection_date), 311 rodent complaints (complaint_date,latitude,longitude), building permits (latitude,longitude,job_start_date,end_date) amd daily weather data (max temp, min temp, total_precip). [download_data](/data_pipeline/download_data.py) [download_weather_data](/data_pipeline/download_bulk_weather.py)
- The for each row in the restaurant inspection, we extract the MM-YYYY. Then, in that month, we find how many rat complaints and building permits were regiestered and in what radius. The radius ranges from 0.1mi to 1.0mi, in 0.1mi increments. We then find the max, min temperature, total precipitation days for that month. The lat long intersection is done using GeoPandas. The resulting data frame, is saved as a csv file. This csv file is then called by the train test val split file. The valiadtion dataset consists of only the rows for that month, the testing data consists of the previous 3 months data, and the training data is all the remaining data. [transform_data](/data_pipeline/transform_data.py)
- *Optional* Data dashboard: We have built a data dashboard that can query the final transformed data. This visualization is done using Grafana. Here, the inspectors can see the historical rat complaints or building permit data by the borough, and month. Additionally, we have added to see the monthly historical data for a given restaurant camis (ID). By looking at this graph, the health inspectors can gain an insight to which areas they can target. [Grafana](http://129.114.25.90:3000/goto/ox7IgMaNR?orgId=1)


## Model training and training platforms
  

The project focuses on predicting rodent infestation in restaurants using a combination of models. We decided to split the problem statement in two parts. 

- In the first model, we predict the probability that a geographical region surrounding the restaurant is infested by rodents. For this we use data from 311 complaints regarding rodents as labels. The number of rodent calls serve as a severity score where higher is worse. This is measured in a geographical region of 0.5 miles around the restaurant. To model this, we used a graph attention network. Graph Networks work particularly well for modeling geographic relations (as neighbour nodes being infested can affect our nodes). This script can be seen here [Graph Model](<Model Training/scripts/gat-ray.py>).

- The second part of the problem is using the temporal dependencies, i.e. using past historic data to predict if the particular restaurant might be infested. For this model, we use past 3 inspections scores and violations, and the predicted infestation score (which we get from the first model). This data is combined to predict if the restaurant might be infested or not. For this, we use a XGBoost classifier. One of the main reasons to use an XGBoost model is its treatment of missing data. Most restaurants do not have more than a couple past inspections and XGBoost can use the rest of the data to make a good prediction. The script to train this can be seen here [XGBoost Model](<Model Training/scripts/restaurant_infestation_predictor-final.py>).

We started with 1 graph model for all the restaurants and 5 xgboost models for each borough. We decided on a model for each borough to boost accuracy as the observed behavior for each borough and hence the test accuracy was quite different. We had to move to 5 graph models (1 for each borough) as the graph which we had to model was too huge to fit one the GPUs. As a result, we moved to 5 graph models - 2 models for each borough. 

To train the models, we had to use **DDP** with **RayTrain** because we got Out of Memory when running without DDP. We also used RayTrain to execute fault tolerance and checkpointing to out object storage.

  We used a single ray worker with 2 GPU nodes, using 2 workers and running the RayTrain Job in DDP. All the models, the params and the metrics were pushed to MLFlow on each run. The Mlflow experiments can be seen here : [MLFlow](<http://129.114.25.90:8000/#/experiments/24?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D>)

For the xgboost models, the train jobs are submitted using ray. We also used **Ray Tune** to find the best config for the XGBoost models. This code can be seen here. [XGBoost Model- ray](<Model Training/scripts/restaurant_infestation_predictor-ray.py>)
. The config from this was used for recurring train. The results for this comparison can be seen in the experiment on MLFlow here.[MLFlow-comparison](<http://129.114.25.90:8000/#/compare-runs?runs=[%2267044be4e666477fa55766f85456e38d%22,%222d9f9f9ed14f4158be59a206234465e5%22,%22a8aa1df5a0b04334938fab7fe233177d%22,%2247296c769da146e7b9d7c197c24d5451%22]&experiments=[%2224%22]>)


The model retrain will be triggered by Argo Workflows  which will trigger an endpoint that runs an Ansible playbook.

We can see a calibration curve below which shows that the more restaurants are infested where we show a higher probability of infestation.![alt text](<calibration_curve.jpg>)

## Model serving and offline evaluation
The steps taken to implement model serving and monitoring platforms are as follows:
- Model serving:
    - After the latest version of the models(1 graph and xgboost per borough) has been registered and tagged for production on MLFlow, a Flask app is used to wrap this configuration of models into an inference service as can be seen here:[Flask App]([pytest suite](https://github.com/adkrish1/Rodent-Infestation-Prediction/blob/main/Model_Evaluation/tests/conftest.py)
        - The input is a choice from the dropdown list of 5 boroughs in NYC: Manhattan, Brooklyn, Queens, Bronx and Staten Island. A user could also access the predict endpoint directly by making the following call:
        ```
        requests.post("http://129.114.25.90/predict", json={"borough": borough})
        ```
        - The output is a table returning the top 10 restaurants in order of highest rodent infestation probability.
    - When considering model optimization techniques, we attempted to compare the runtimes of eager vs compile mode of the graph model which did not lead to a significant improvement and hence was discarded. For the case of xgboost, it is a tree based model already optimized for inference.
    - For our offline evaluation, we have a pytest suite that can be found here:[pytest suite](https://github.com/adkrish1/Rodent-Infestation-Prediction/blob/main/Model_Evaluation/tests/conftest.py)
    - The results of this pytest our logged in mlflow. For e.g., [pytest on MLFlow example](http://129.114.25.90:8000/#/experiments/25/runs/85bc40f2a2aa41d7b41156179008f991)
    - We created a load test file that we manually trigger and monitor on the grafana dashboards.
    - A business specific evaluation in this case would be to compare how many restaurants from the top 10 list of each borough actually had infestations reported in them for the next month. This will help us know how far off from the actual values we are.

## Model and Data Online Monitoring
- [Rodent Infestation Model Dashboard](http://129.114.25.90:3000/goto/EH-QkMaHR?orgId=1)
- [Rodent Infestation Service Monitoring](http://129.114.25.90:3000/goto/RzF_kMaNg?orgId=1)
- **EXTRA DIFFICULTY** [Online Data Dashboard](http://localhost:3000/goto/Mjs-zG-Hg?orgId=1)

## Staged deployment: 

We use Argo Workflows run through ansible playbooks to deploy our trained models canary, staging and production.
- Ansible: https://github.com/adkrish1/Rodent-Infestation-Prediction/tree/main/Continous_X/ansible/argocd

## CI/CD and continuous training:
An Argo Workflow runs a python script to update data weekly. We use ArgoCD workflows to trigger the training pipeline. ArgoCD triggers an external Flask endpoint that runs an ansible playbook to trigger training on a GPU. Once the training completes we run evaluation pytests and then push tag the model to staging. We use the promote-models workflow template to promote models from staging to canary and canary to production. Model version is correctly updated in these workflow templates. 

- Flask: 
- Workflows: https://github.com/adkrish1/Rodent-Infestation-Prediction/tree/main/Continous_X/workflows


## Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

|                                             | How it was created                                         | Conditions of use                                         | Links    |
|---------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|----------|
| Rat Sightings                               | 311 Service Requests                                       | Public domain                                             | [Link](https://data.cityofnewyork.us/Social-Services/Rat-Sightings/3q43-55fe/about_data)
| Restaurant Inspection Results               | Department of Health and Mental Hygiene (DOHMH)            | Public domain                                             | [Link](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/about_data)
| Meteostat Developers Climate Data                           |  NOAA and DWD        | CC BY-NC 4.0  | [Link](https://dev.meteostat.net/)
| DOB Permit Issuance           |Department of Buildings (DOB)                               | Public domain                                             | [Link](https://data.cityofnewyork.us/Housing-Development/DOB-Permit-Issuance/ipu4-2q9a/about_data)
| GAT: Graph Attention Network| Petar Veličković | MIT License | [Paper](https://arxiv.org/pdf/1710.10903v3) [GitHub](https://github.com/PetarV-/GAT)
| XGBoost | https://github.com/dmlc/xgboost |  Apache-2.0 license | [Link](https://xgboost.readthedocs.io/en/release_3.0.0/#)
