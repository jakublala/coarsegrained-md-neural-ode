import torch
import numpy as np
import sigopt
import os

from diffmd.training import Trainer

def run_and_track_in_sigopt():
    #   sigopt.log_dataset(DATASET_NAME)
    #   sigopt.log_metadata(key="Dataset Source", value=DATASET_SRC)
    #   sigopt.log_metadata(key="Feature Eng Pipeline Name", value=FEATURE_ENG_PIPELINE_NAME)
    #   sigopt.log_metadata(
    #     key="Dataset Rows", value=features.shape[0]
    #   )  # assumes features X are like a numpy array with shape
    #   sigopt.log_metadata(key="Dataset Columns", value=features.shape[1])
    #   sigopt.log_metadata(key="Execution Environment", value="Colab Notebook")
    
    sigopt.log_model('CG Hexagon Potential - Fourth Search (Small NN)')
    os.environ["SIGOPT_PROJECT"] = "coarsegrained-md-neural-ode"
    os.environ["sigopt_project_id"] = "coarsegrained-md-neural-ode"
    os.environ["SIGOPT_PROJECT_ID"] = "coarsegrained-md-neural-ode"
    # learning_rates = [10**i for i in range(-6, 2)]
    # sigopt.params.setdefaults(
    #     # batch_length=np.random.randint(low=3, high=50),
    #     # nbatches=np.random.randint(low=10, high=1000),
    #     learning_rate=np.random.choice(learning_rates),
    #     nn_depth=np.random.randint(low=1, high=5),
    #     nn_width=np.random.randint(low=2, high=1000),
    #     # activation_function=,  
    # )

    # TODO: add a script that creates experiment.yml based on config OR use this experiment function
    # experiment = sigopt.create_experiment(
    # name="Keras Model Optimization (Python)",
    # type="offline",
    # parameters=[
    #     dict(name="hidden_layer_size", type="int", bounds=dict(min=32, max=128)),
    #     dict(name="activation_function", type="categorical", categorical_values=["relu", "tanh"]),
    # ],
    # metrics=[dict(name="holdout_accuracy", objective="maximize")],
    # parallel_bandwidth=1,
    # budget=30,
    # )

    config = dict(
        folder = 'dataset/repulsive/', 
        device = torch.device("cuda"), 
        start_niter=0,
        niters = 20000,
        optimizer = 'Adam',
        batch_length=sigopt.params.batch_length,
        nbatches=sigopt.params.nbatches,
        learning_rate=np.log(sigopt.params.learning_rate),
        nn_depth=sigopt.params.nn_depth,
        nn_width=sigopt.params.nn_width,
        activation_function=None,
        load_folder=None,
        dtype=torch.float32,
        printing_freq=500,
        plotting_freq=21000,
        stopping_freq=1000,
        scheduler='LambdaLR',
        scheduling_factor=sigopt.params.scheduling_factor,
        scheduling_freq=sigopt.params.scheduling_freq,
        evaluation_freq=500,
    )

    # sigopt.log_dataset(dataset) 
    
    trainer = Trainer(config)
    model, train_loss = trainer.train()
    trainer.save()
    eval_loss = trainer.evaluate(training_dataset=True)

    # running_avg_train_loss = train_loss.avg
    # current_train_loss = train_loss.val

    # sigopt.log_metric(name="train_loss", value=current_train_loss)
    sigopt.log_metric(name="train_loss", value=eval_loss)
    # sigopt.log_metric(name="test_loss", value=running_avg_test_loss)
    # sigopt.log_metric(name="training time (s)", value=traininx    g_time)
    # sigopt.log_metric(name="training and validation time (s)", value=training_and_validation_time)

run_and_track_in_sigopt()
# RUN sigopt optimize -e experiment.yml python sigopt-model.py > sigopt-model.out