\# Structural Damage Recognition



Machine Learning mini-project for binary classification of structural images into:



\- Damaged

\- Undamaged



This project is based on the assigned Structural Damage Recognition problem and uses the official PEER Φ-Net Task 2 — Damage State dataset.



\---



\## 1. Project Overview



The objective of this project is to develop a machine learning system that can classify an input structural image as either damaged or undamaged.



The project is being developed as part of the UE24CS352A Machine Learning Mini-Project.



The project will include:



1\. Dataset analysis

2\. Image visualization

3\. Data preprocessing

4\. Baseline machine learning/deep learning model

5\. Transfer-learning based model

6\. Model evaluation

7\. Error analysis

8\. Live prediction demonstration



\---



\## 2. Reference Work



The assigned reference work is:



\*\*Structural Damage Image Classification\*\*



CS229 Project Report, 2018.



Reference paper:



https://cs229.stanford.edu/proj2018/report/39.pdf



Reference poster:



https://cs229.stanford.edu/proj2018/poster/39.pdf



The original reference work investigates structural damage classification using machine learning and deep learning approaches.



\---



\## 3. Dataset



\### Source



Official PEER Φ-Net dataset:



https://apps.peer.berkeley.edu/phi-net/



Task:



\*\*Task 2 — Damage State\*\*



The dataset contains structural images belonging to two classes:



\- Undamaged

\- Damaged



\### Dataset files



The downloaded dataset is stored locally under:



```text

DATA/

├── task2\_damage\_state\_1/

│   ├── task2\_X\_test.npy

│   ├── task2\_y\_test.npy

│   └── task2\_y\_train.npy

│

└── task2\_damage\_state\_2/

&#x20;   └── task2\_X\_train.npy

