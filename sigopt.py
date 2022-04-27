import torch
import numpy as np
import sigopt

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
    sigopt.log_model('CG Hexagon Potential - First Search')
    # TODO: add a script that creates experiment.yml based on config
    learning_rates = [10**i for i in range(-5, 1)]
    sigopt.params.setdefaults(
        batch_length=np.random.randint(low=3, high=50),
        nbatches=np.random.randint(low=10, high=1000),
        learning_rate=np.random.choice(learning_rates),
        nn_depth=np.random.randint(low=1, high=5),
        nn_width=np.random.randint(low=2, high=50),
        # activation_function=,  
    )

    prefix = 'hexagons/trajectories/smooth/'
    config = dict(
        filename = prefix+'NVE-temp-0.45_K-0.090702947845805_r-0_s-5', 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        niters = 100,
        optimizer = 'Adam',
        batch_length=sigopt.params.sample_length,
        nbatches=sigopt.params.batch_size,
        learning_rate=sigopt.params.learning_rate,
        nn_depth=sigopt.params.nn_depth,
        nn_width=sigopt.params.nn_width,
        activation_function=None,
    )

    trainer = Trainer(config)
    model, train_loss = trainer.train()

    running_avg_train_loss = train_loss.avg

    sigopt.log_metric(name="train_loss", value=running_avg_train_loss)
    # sigopt.log_metric(name="test_loss", value=running_avg_test_loss)
    # sigopt.log_metric(name="training time (s)", value=training_time)
    # sigopt.log_metric(name="training and validation time (s)", value=training_and_validation_time)

if __name__ == "__main__":
    run_and_track_in_sigopt()