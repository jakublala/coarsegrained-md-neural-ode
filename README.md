# Coarse-Graining of Molecular Dynamics Using Neural ODEs
Jakub Lala's thesis repository on neural ordinary differential equations used for coarse-graining molecular dynamics.
some intro

## Table of Contents
[Theoretical Background](#theory) <br>
[Project's Vision](#vision) <br>
[Code Structure](#code) <br>
[How to Use](#howto) <br>
[Collaboration](#collab) <br>


## Theoretical Background <a name="theory"></a>

## Project's Vision <a name="vision"></a>

## Code Structure <a name="code"></a>

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
** `all`
** `final`

* itr_printing_freq=1,
* printing_freq=20,
* plotting_freq=20,
* stopping_freq=20,
* scheduling_freq=25,
* evaluation_freq=20,
* checkpoint_freq=20,






## Collaboration <a name="collab"></a>
If you are eager to help out with this project, I am more than happy to get you on board. There is a lot of small fixes and optimization things that could be improved. 

Ideally you should create your own fork and then for each feature added make a new branch. Once you are happy for review, you can submit a pull request and we can discuss the changes/improvements. Either way, you can contact me at <a href="mailto:jakublala@gmail.com">jakublala@gmail.com</a> so that we can be in touch and figure out the details. 
