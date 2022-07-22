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

* folder = 'dataset/single_temp_overfit', 
* load_folder=None,
* device = torch.device('cpu'), 
* dtype=torch.float32,

* epochs = 100,
* start_epoch = 0,
* nn_depth=2,
* nn_width=1000,
* batch_length=20,
* eval_batch_length=1000,
* batch_size=600,
* shuffle=True,
* num_workers=0,
* learning_rate=0.0003,

* scheduler='LambdaLR',
* scheduling_factor=0.95,
* optimizer = 'Adam',
* loss_func = 'all',

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
