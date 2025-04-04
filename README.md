
## Title of project

<!-- 
Discuss: Value proposition: Your will propose a machine learning system that can be 
used in an existing business or service. (You should not propose a system in which 
a new business or service would be developed around the machine learning system.) 
Describe the value proposition for the machine learning system. What’s the (non-ML) 
status quo used in the business or service? What business metric are you going to be 
judged on? (Note that the “service” does not have to be for general users; you can 
propose a system for a science problem, for example.)
-->

### Contributors

<!-- Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions. If your project involves multiple repos, you will 
link to their contributions in all repos here. -->

| Name                            | Responsible for | Link to their commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| All team members                |                 |                                    |
| Team member 1                   |                 |                                    |
| Team member 2                   |                 |                                    |
| Team member 3                   |                 |                                    |
| Team member 4 (if there is one) |                 |                                    |



### System diagram

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

|              | How it was created | Conditions of use |
|--------------|--------------------|-------------------|
| Data set 1   |                    |                   |
| Data set 2   |                    |                   |
| Base model 1 |                    |                   |
| etc          |                    |                   |


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 3 for entire project duration                     | ...           |
| `gpu_mi100`     | 4 hour block twice a week                         |               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| etc             |                                                   |               |

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Model serving and monitoring platforms

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

#### Continuous X
To ensure the model remains accurate and effective over time, we implement a **CI/CD/CT pipeline** that automates training, evaluation, deployment, and monitoring. The pipeline integrates modern DevOps tools such as **Argo Workflows, Helm, GitHub Actions, MLflow, and Kubernetes** to ensure seamless operation in a cloud-native environment.

### **Continuous Integration (CI)**

#### *Version Control and Automation*

* **GitHub** is used for version control, where all model code, infrastructure configurations (Infrastructure-as-Code), and deployment scripts are maintained.  
* **GitHub Actions** automates code quality checks and unit tests for data and model pre-processing.

### **Continuous Training (CT)**

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

**Relation to Lecture Material**  
*As per Unit 3:*

* **Infrastructure-as-Code:** We use **Helm, ArgoCD, and GitHub** to define and manage infrastructure declaratively, avoiding manual configurations.

* **Cloud-Native:** The system follows **immutable infrastructure**, **microservices (FastAPI model server)**, and **containerized deployments (Docker \+ Kubernetes)**.

* **CI/CD & Continuous Training:** **Argo Workflows** automates model retraining, **MLflow** tracks performance, and **GitHub Actions** ensures code quality.

* **Staged Deployment:** **Helm & ArgoCD** manage deployments across **staging, canary, and production**, ensuring safe rollouts.  