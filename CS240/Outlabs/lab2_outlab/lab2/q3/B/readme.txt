You have fill up the TODO sections in the code files as per the instructions given in the context.

In the algos folder,
- LSLR_algo1.py
- LSLR_algo2.py
- LSLR_algo3.py

Refer to LSLROptimiser base class in optim.py for reference while implementing these algorithms.
These are the algorithms that will be used to evaluate performance in Task B.

Note that B.py is only for your own experimentation and will not be used in grading. 
We will be importing the LSLRAlgo1, LSLRAlgo2 and LSLRAlgo3 classes from the algos folder directly in our grading scripts.

Running B.py instructions - 

Usage: 
    python B.py --data_folder <path_to_data_folder> --tolerance <tolerance_value> --max_epochs <max_epochs_value>   

for example

    python B.py --data_folder ../data/datasets/D/ --tolerance 1e-8 --max_epochs 10000


Use, 1e-8 as tolerance for D and 1e-2 as tolerance for A.

Also B runs your algorithms until loss value reaches to the minimum possible loss upto the specified tolerance.

Also you can refer to ||w_t - w_t-1||^2 v/s epochs plot for debugging your algorithms.

Also, you can fill algorithm names in place of <algo1_name>, <algo2_name> and <algo3_name> in B.py for better readability of outputs.


Also w_init = np.zeros((n, 1)) is used for all algorithms as initial weights.


