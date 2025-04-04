
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

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->

### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->


