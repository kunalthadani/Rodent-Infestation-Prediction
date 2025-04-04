
# Rodent Infestation Prediction

<!-- 
Discuss: Value proposition: Your will propose a machine learning system that can be 
used in an existing business or service. (You should not propose a system in which 
a new business or service would be developed around the machine learning system.) 
Describe the value proposition for the machine learning system. What’s the (non-ML) 
status quo used in the business or service? What business metric are you going to be 
judged on? (Note that the “service” does not have to be for general users; you can 
propose a system for a science problem, for example.)
-->

## Value proposition

### NYC Department of Sanitation
- Existing business model: Manages waste collection and disposal to maintain public cleanliness, issues fines for improper garbage disposal, runs pests/rodent mitigation programs like rat-proof trash bins and waste management policies 
- Our value proposition: optimized waste collection to reduce rodent infestation, targeted cleanup in high-risk areas, runs rodent mitigation programs like rat-proof trash bins and waste management policies

### NYC Housing Authority
- Existing business model: manages public housing for low-income residents, handles maintenance and pest control within its properties, responds to resident complaints about infestations.
- Our value proposition: proactive pest control, reduced infrastructure damage, costs savings as prevention is cheaper than cure

### Insurance Companies (Insurent, Rhino, etc.)
- Existing business model: assesses risks related to property damage and liability, offers policies that may or may not cover rodent damage.
- Our value proposition: better risk assessment (adjusts property insurance pricing based on rodent activity data), new policy offerings (could offer pest insurance to high-risk areas), fewer payouts (encourages policyholders to take preventive actions, reducing claims)

### Pest Control Companies
- Existing business model: provides extermination and pest prevention services, operates reactively based on customer complaints
- Our value proposition: predictive pest control Services (can offer preemptive treatment plans before infestations start), better resource allocation (deploys technicians where they’re needed most), increased revenue (subscription-based prevention services could be introduced)

---

### Contributors

<!-- Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions. If your project involves multiple repos, you will 
link to their contributions in all repos here. -->

| Name                            | Responsible for                      | Link to their commits in this repo |
|---------------------------------|--------------------------------------|------------------------------------|
| Aditya Krishna                  |Model serving and monitoring platforms|[Aditya's commits](https://github.com/adkrish1/Rodent-Infestation-Prediction/commits/main/?author=adkrish1)|
| Akshay Hemant Paralikar         |Model training and training platforms |[Akshay's commits](https://github.com/adkrish1/Rodent-Infestation-Prediction/commits/main/?author=akshay412)|
| Kunal Thadani                   |Continious X                          |[Kunal's commits](https://github.com/adkrish1/Rodent-Infestation-Prediction/commits/main/?author=kunalthadani)|
| Rakshith Murthy                 |Data Pipeline                         |[Rakshith's commits](https://github.com/adkrish1/Rodent-Infestation-Prediction/commits/main/?author=valar007)|

### System diagram

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->

![System Design Diagram](./System%20Design.png)

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

|                                             | How it was created                                         | Conditions of use                                         | Links    |
|---------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|----------|
| 2020 Neighborhood Tabulation Areas (NTAs)   | Department of City Planning (DCP)                          | Public domain                                             | [Link](https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-/9nt8-h7nd/about_data)
| 311 Rodent Complaints                       | Subset of 311 complaints by Louis DeBellis on NYC Open Data| Public domain                                             | [Link](https://data.cityofnewyork.us/Social-Services/311-Rodent-Complaints/cvf2-zn8s/about_data)
| Restaurant Inspection Results               | Department of Health and Mental Hygiene (DOHMH)            | Public domain                                             | [Link](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/about_data)
| NOAA Climate Data                           | NOAA National Centers for Environmental Information        | FAIR (Findable, Accessible, Interoperable, and Reusable)  | [Link](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00861/html)
| Garbage Collection Frequencies              | Department of Sanitation                                   | Public domain                                             | [Link](https://data.cityofnewyork.us/City-Government/DSNY-Frequencies/rv63-53db/about_data)
| Monthly Monthly Tonnage Data                | Department of Sanitation                                   | Public domain                                             | [Link](https://data.cityofnewyork.us/City-Government/DSNY-Monthly-Tonnage-Data/ebb7-mvp5/about_data)
|A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting| Jiawei Zhu | Open Access CC BY 4.0 | [Paper](https://arxiv.org/pdf/2006.11583) [GitHub](https://github.com/lehaifeng/T-GCN)



### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| 3x `m1.medium` VMs | For entire project duration                     | One for data pipeline, one for MLFlow and one for model serving           |
| 2x `gpu_A100`     | A 4 hour block twice a week               | Development and training of the model. A100 specifically because the training size and time of a TGNN scales with increase in data size               |
| 2x Floating IPs    |For entire project duration | 1 for FastAPI endpoint and 1 for MLFlow and internal Grafana Dashboard              |
| 1x `gpu_v100` or less powerful |A 4 hour block every week                                       |Will be required for model serving(inference testing)           |
| Persistent Store            |  30 GiB                                                  | All data stores amount to about 10-15 GB, continuously storing all models and docker containers will require about ~10 GiB               |

## Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

### Model training and training platforms

#### Strategy:

The project will focus on predicting rodent infestation hotspots using Temporal Graph Attention Network. Graph temporal neural networks are well-suited for this as they model both spatial and temporal dependencies of their neighbors. We will model changes in environmental conditions,  sanitation patterns(for e.g. Restaurant health inspection data and garbage collection data) and historical complaints to predict future infestation behaviours. We will gauge the performance of the model by holding out some data and comparing the severity of infestation predicted. The model will be trained and retrained on a fixed timeframe(every week) schedule to update recent changes in the data. To ensure scalability and efficiency, distributed training and hyperparameter tuning will be implemented using Ray Train and Ray Tune. Additionally, model versioning and artifact storage will be integrated into the pipeline to manage different model iterations effectively. 

#### Relevant Parts of the Diagram:

Model Architecture: Temporal Graph Network (TGN) with temporal embeddings to capture dynamic interactions.
Distributed Training Setup: A Ray cluster with multiple GPUs for parallel training.
Experiment Tracking and model versioning: MLflow for logging metrics, hyperparameters, and artifacts.
Checkpointing & Fault Tolerance: Ray Train's built-in checkpointing and fault tolerance mechanisms to ensure recovery from failures.
Hyperparameter Tuning: Ray Tune integrated with W&B for efficient hyperparameter 
Optimization.



#### Justification for Strategy

Temporal Graph Attention Networks: Ideal for dynamic graph-based problems which have both spatial and temporal connections. Graph can be scaled as per data granularity

Ray Train for Distributed Training: Enables scaling across multiple GPUs or nodes while providing fault tolerance through checkpointing ensuring minimal disruption in case of node or worker failures. 

Ray Tune with W&B Integration: Combines the efficiency of Ray Tune's advanced search algorithms (e.g., HyperBand) with real-time monitoring of hyperparameter tuning experiments.

Model Versioning & Artifact Storage: Storing models as artifacts in MLFlow ensures iterations are preserved for comparison, deployment, or rollback if needed.

#### Relating to lecture material

Unit 4 : We will retrain the model every week to update the model with recent data.This will be submitted as a training job as part of the pipeline.

Unit 5:
Similar to the experiments run in Unit 5 for model training with MLFlow and Ray, we will use Ray Clusters for checkpointing ensuring minimal disruption and MLFlow for versioning the model. 

#### Specific Numbers:

The model size will depend on the granularity of geographic data we choose. The data will be modelled at a chosen granularity-week(space-temporal) level where each node will represent a geographical block based on the granularity selected.

New York City can be divided into ~250 neighborhoods which can be nodes in the graph. On a weekly level, we have 52 * 5 = 260 weeks of data. 
Each node will have subsequent edges with its neighbours and additional temporal edges with nodes for the next week.

To train this model, we will ideally need  2X A100 GPUs twice a week for about 3-4 hours during development and can move to once a week(time required will be known based on the development training experiments) during actual deployment.



### Model serving and monitoring platforms
The steps that will be taken to implement model serving and monitoring platforms are as follows:
- Model serving:
    - After the latest version of the model has been stored as an artifact by MLFlow, FastAPI will be used to wrap the artifact into a standalone inference service as can be seen in the system design diagram, this is in reference to unit 6 as a part of Lab Part 3.
    - As a part of development, we will perform benchmarking tests to find the optimal system and model optimizations specifically for serving, aiming to achieve an inference time of about 10-15 seconds, this is in reference to unit 6 specifically covering Lab Part 1 and 3.
    - We will also have a frontend deployed that will serve as the user interface. This UI will let the user select a geographical block, the granularity of which will be decided upon experimentation during development, from a drop-down list/search bar and also a specific duration of time for which the severity of rodent infestation needs to be calculated, hence justifying the need for FastAPI.
    - The inference will return both a severity score and a link to the grafana dashboard for further visualization.
    - (Extra Difficulty Points) Developing multiple options for inference servers, especially server-grade CPU and GPU will be attempted.
- Model Monitoring:
    - As soon as the latest artifact is ready and wrapped in FastAPI, a series of offline tests, as per unit 7 will be performed through the help of the continous X pipeline as follows:
        - A sanity check is performed to make sure the system is working normally from a general standpoint.
        - A unit testing will then be performed to test optimizing, operational and behavioural metrics, especially metrics like accuracy, loss and inference time. 
        - All unit tests should pass or else the tested build version is considered to have failed and an alert will be sent to prometheus which will then be displayed on the internal grafana dashboard. 
        - Both of these services as well as the flow can be seen in the system design diagram.
    - A load test in staging is performed after all the offline tests are passed.
    - We then perform an online canary test, as per the slides in unit 7.

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->

### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

- Persistant storage: 30GB of storage on Chameleon Object Store to store the models (storing 30 previous models), datasets (10-15GB), docker containers (5GB), etc
- Offline data that is used for training is stored in the persistant storage. The data will be split into training, testing and validation. Some of the validation data will be saved for production data, that will be used to test the model after deployment in production
- The datasets are updated daily, we are running a Python script to download the newer data and transform it
- Since the NYC datasets do not contain the neighbourhood data, we will be transforming the data to include that using geo-spacial data and lat-long co-ordinates for inferencing
- For simulating the real world data, we require how far along does the end user require the prediction (1 week, 1 month, 2 weeks, etc.), lat-long co-ordinates or a geographical block (will depend on granularity chosen)
- [Difficulty point] During the ETL pipeline, metadata (number of new instances, any missing data, errors, etc) of the data retrieved for training will be sent to Prometheus. The high level view of this will be queryable in Grafana for the team members


### Continuous X
To ensure the model remains accurate and effective over time, we implement a CI/CD/CT pipeline that automates training, evaluation, deployment, and monitoring. The pipeline integrates modern DevOps tools such as Argo Workflows, Helm, GitHub Actions, MLflow, and Kubernetes to ensure seamless operation in a cloud-native environment.

**Continuous Integration (CI)**
#### *Version Control and Automation*
* **GitHub** is used for version control, where all model code, infrastructure configurations (Infrastructure-as-Code), and deployment scripts are maintained.  
* **GitHub Actions** automates code quality checks and unit tests for data and model pre-processing.

**Continuous Training (CT)**
#### *Automated Model Retraining Workflow*
* The model needs to adapt to changing environmental and sanitation conditions.  
* A weekly scheduled job in **Argo Workflows** retrains the model with the latest data.  
* The workflow job is:  
1. Loads the latest dataset from **Chameleon persistent storage**.  
2. Preprocesses and engineers features.  
3. Trains the model.  
4. Logs the new model and its metrics in **MLflow**.  
5. Runs an offline evaluation against the previous model.  
6. If performance surpasses a predefined threshold, the model is registered for deployment.  
* If a significant drift in model performance is detected:  
1. A retraining job is triggered automatically.  
2. The new model is evaluated against the previous version.  
3. If the new model outperforms the old one, it is deployed following the CX pipeline.

**Continuous Deployment (CD)**
#### *Containerization and Deployment*
* The trained model is packaged as a **FastAPI** service, exposing REST endpoints for predictions.  
* **Docker** is used to containerize the FastAPI service.

#### *Deployment to Kubernetes*
* **Helm** manages the deployment of the FastAPI model server to a **Kubernetes** cluster.

#### *ArgoCD for Continuous Deployment*
* **ArgoCD** monitors the Git repository for new versions of the **Helm charts** and automatically updates Kubernetes deployments.

#### Relation to Lecture Material 
*As per Unit 3:*
* **Infrastructure-as-Code:** We use **Helm, ArgoCD, and GitHub** to define and manage infrastructure declaratively, avoiding manual configurations.
* **Cloud-Native:** The system follows **immutable infrastructure**, **microservices (FastAPI model server)**, and **containerized deployments (Docker \+ Kubernetes)**.
* **CI/CD & Continuous Training:** **Argo Workflows** automates model retraining, **MLflow** tracks performance, and **GitHub Actions** ensures code quality.
* **Staged Deployment:** **Helm & ArgoCD** manage deployments across **staging, canary, and production**, ensuring safe rollouts. 