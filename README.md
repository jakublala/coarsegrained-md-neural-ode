# Coarse-Graining of Molecular Dynamics Using Neural ODEs
This project is done by Jakub Lála under the supervision of [Stefano Angiolleti-Uberti](https://www.imperial.ac.uk/people/s.angioletti-uberti) in the [SoftNanoLab](https://www.imperial.ac.uk/people/s.angioletti-uberti) at Imperial College London. It started off as a part of my Master's thesis, but the research and development is currently ongoing. We utilise the state-of-the-art deep learning method of neural ordinary differential equations ([neural ODEs](https://github.com/rtqichen/torchdiffeq)) to learn coarse-grained (CG) machine learning (ML) potentials for any molecule, nanoparticle, etc. By leveraging the idea of updating many parameters at once when one learns on a dynamical trajectory rather than frozen time snapshots of configuration-energy pairs, we aim to develop an automated coarse-graining pipeline that produces computationally cheaper ML potentials, compared to running all-particle simulations of complex molecules and nanoparticles at the atomistic resolution. 

## Theoretical Background <a name="theory"></a>
By considering a complex, composite body made up of many particles as a rigid body with a single centre of mass and an orientation, one tremendously reduces the amount of degrees of freedom that need to be simulated, thus theoretically decreasing the computational demand.
![github_test](https://user-images.githubusercontent.com/68380659/187448405-abab9a05-f5fc-41e7-886a-a182c74a6f25.png)

## How to Use <a name="howto"></a>
To run the code, one has to first get comfortable with the `Trainer` class that forms the basis of all training, including testing and validating. It takes a `config` dictionary input that consists of the following elements:

* `folder`: relative path to the folder with the datasets 
* `load_folder`: relative path to a pre-trained `model.pt` file 
* `device`: device to train on 
* `dtype`: datatype for all tensors

* `epochs`: number of training epochs
* `start_epoch`: number of the starting epoch (useful when re-training a model)
* `nn_depth`: depth of the neural net
* `nn_width`: width of the neural net
* `batch_length`: trajectory length used for training
* `eval_batch_length`: trajectory length used for model performance evaluation
* `batch_size`: number of trajectories in a single batch
* `shuffle`: if set to `True`, the dataloader shuffles the trajectory order in the dataset during training
* `num_workers`: number of workers used by dataloader
* `optimizer` = name of the optimizer (e.g. `Adam`)
* `learning_rate`: initial learning rate

* `scheduler`: name of the scheduler (e.g. `LambdaLR`)
* `scheduling_factor`: scheduling factor determining the rate of scheduling
* `loss_func`: type of loss function
  - `all`: absolute mean difference of the entire trajectory
  - `final`: absolute mean difference of the final state in the trajectory

* `itr_printing_freq`: frequency of printing for iterations in an epoch
* `printing_freq`: frequency of printing for epochs
* `plotting_freq`: frequency of plotting for epochs
* `stopping_freq`: frequency of early stopping for epochs (e.g. due to non-convergent loss)
* `scheduling_freq`: frequency of scheduling the learning rate for epochs
* `evaluation_freq`: frequency of evaluating the model on the test dataset
* `checkpoint_freq`: frequency of saving a checkpoint of the model

For an example run file look at [run-example.py](run-example.py).



## Collaboration <a name="collab"></a>
If you are eager to help out with this project, I am more than happy to get you on board. There is a lot of small fixes and optimization things that could be improved. Also, if you want to test out this code on your own simulations, I am excited to help you out, as it would also allow us to properly benchmark and test out this method

Ideally you should create your own fork and then for each feature added make a new branch. Once you are happy for review, you can submit a pull request and we can discuss the changes/improvements. Either way, you can contact me at <a href="mailto:jakublala@gmail.com">jakublala@gmail.com</a> so that we can be in touch and figure out the details. 
