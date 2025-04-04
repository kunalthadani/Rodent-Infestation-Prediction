
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
- Existing business model: Manages public housing for low-income residents, handles maintenance and pest control within its properties, responds to resident complaints about infestations.
- Our value proposition: proactive pest control, reduced infrastructure damage, costs savings as prevention is cheaper than cure

### Insurance Companies (Insurent, Rhino, etc.)
- Existing business model: Assesses risks related to property damage and liability, offers policies that may or may not cover rodent damage.
- Our value proposition: Better Risk Assessment (adjusts property insurance pricing based on rodent activity data,) new Policy Offerings (could offer pest insurance to high-risk areas), fewer Payouts (encourages policyholders to take preventive actions, reducing claims)

### Pest Control Companies
- Existing business model: provides extermination and pest prevention services, operates reactively based on customer complaints
- Our value proposition: predictive pest control Services (can offer preemptive treatment plans before infestations start), better Resource Allocation (deploys technicians where they’re needed most), increased revenue (subscription-based prevention services could be introduced)

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

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->


