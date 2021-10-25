# Quantum machine learning beyond kernel methods

This git repository accompanies the pre-print "Quantum machine learning beyond kernel methods" [arXiv:2110.](https://arxiv.org/abs/2110.) by providing the code used to run its numerical simulations, along with their resulting data.

# Runfiles

``` main.py ```

Main python script, containing the code to generate the explicit and implicit models, the pre-processed fashion MNIST datasets, and to train the quantum models on the resulting task.

Takes as arguments: system size, number of layers in the variational circuit, number of training points, experiment number, a Boolean specifying whether to use the 1D-Heisenberg variational evolution as a variational circuit. 

Stores a pickle file in ```./results/``` containing: the generating explicit model storing its dataset, the trained explicit model storing its learning history, the implicit model (storing the kernel matrix, for experiment number 0), and the training and validation errors of the implicit model for regularized and unregularized losses. 

To reproduce the results of Figure 5: run command lines of the form <br/> `python main.py {2->12} {15-10-7-4-4-3-3-3-3-3-2} 1000 {0->9} False` <br/>
e.g., `python main.py 10 3 1000 0 False`
<br/>
<br/>

``` classical.py ```

Python script to run a list of classical learning algorithms on the generated learning tasks. <!> Needs to be executed after ```main.py``` (using the same arguments).

Adds the best classical learning performance to the pickle files above.

To reproduce the results of Figure 5: run command lines of the form <br/> `python classical.py {2->12} {15-10-7-4-4-3-3-3-3-3-2} 1000 {0->9} False` <br/>
e.g., `python classical.py 10 3 1000 0 False`
<br/>
<br/>

``` new_run.py ```

Python script to train a different explicit model on an already generated learning task. <!> Needs to be executed after ```main.py``` (using the same arguments + an additional Boolean specifying whether to use the 1D-Heisenberg variational evolution as a variational circuit for the new explicit model).

Adds the newly trained explicit model (storing its learning history) to the pickle files above.

To reproduce the results of Figure 12: run command lines of the form <br/> `python new_run.py {2->12} {15-10-7-4-4-3-3-3-3-3-2} 1000 {0->9} False True` <br/>
e.g., `python new_run.py 10 3 1000 0 False True`
