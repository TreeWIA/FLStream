# FLstreamline
Fedavg streamline code for 'An Impact Study of Concept Drift in Federated Learning"  

## Dependencies
    copy
    math
    time
    numpy
    arff
    torch
    csv
    torchvision
    matplotlib
    sklearn
    pandas
    random


## How to run
- When running the artificial dataset, you need to modify the drift_type in the main function, and then determine whether the file path of MyDataset is correct;
- When running the real dataset, just modify the file information of MyDataset;
- The default code is to run the forecast covtype dataset.

    


## Scenario Design
- We have established 15 experimental scenarios, encompassing three forms of drift, speed, severity, coverage, and synchronism. 
- The detailed settings inside the scenarios are:
- Form: it is investigated by simulating P (x), P (y),P (y|x) drifts using the Sine generator. For the P (x) change, the old setting generates two classes of data far away from the SINE boundary; the new setting moves data close to       the boundary. For the P (y) change, the old setting is class imbalanced, with a ratio of 9:1 between class 0 and class 1; the new setting is class balanced with a 1:1 ratio between the two classes. For the P (y|x) change, the old         setting labels data above the sine boundary as class 0, and data below the boundary are labeled as class 1. The new setting switches around the two classes.
- Speed: three changing speeds are considered – abrupt, gradual, incremental. The abrupt drift is achieved by abruptly rotating the hyperplane boundary by 180 degrees at time step 200. The gradual drift is a probabilistic change of       180 degrees between time steps 200 and 300, where the probability of the old concept reduces by 10% at every 10 batches. From time step 300, only new concepts are present. For the incremental drift, the
    boundary of the data rotates 180 degrees slowly between time steps 200 and 300.
- Severity: three levels of severity are compared – high, medium, low. A high severity means that the hyperplane rotates 180 degrees – the two labels are interchanged. The medium and low severity has the hyperplane rotate 120 and 60      degrees respectively.
- Coverage: three levels of coverage are considered by varying the number of clients affected by concept drift out of 10 clients – 10/10 (high), 5/10 (medium), 1/10 (low).
- Synchronism: two scenarios are compared – synchronous, asynchronous. The synchronous case has all the clients start and end a concept drift at time step 200. The asynchronous case has the same drift affect clients 2 to 10               sequentially from time step 100 and one after another at every 100 time steps.

## Experimental Settings
All the experiments are implemented in PyTorch. The platform is an Lenovo R9000P with an AMD Ryzen 7 5800H CPU, a NVIDIA GeForce RTX 3070 Laptop GPU, and 16-GB RAM under Windows 11. 
