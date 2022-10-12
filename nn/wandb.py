import wandb

class Wandb():
    def __init__(self, config):
        self.config = config
        self.run = wandb.init(project='diffmd', config=config)
        
    # def init(self, config):

        # weights and biases - logging
        # if self.is_master():
        #     self.run = wandb.init(project="my-test-project",
        #     config={
        #         "epochs": self.epochs,
        #         "batch_size": self.batch_size,
        #         "nn_widths": self.nn_widths,
        #         'n_parameters': self.nparameters,
        #         "activation_function": config['activation_function'],
                
        #         "dataset": self.folder,
        #         "training_fraction": config['training_fraction'],
        #         "random_dataset": config['random_dataset'],

        #         # TODO: currently updating config is not supported
        #         "dataset_steps": self.dataset_steps,
        #         "steps_per_dt": self.steps_per_dt,
        #         # "lammps_dt": self.training_dataset.trajs[0].lammps_dt,
        #         # "logged_dt": self.training_dataset.trajs[0].logged_dt,
        #         # "ratio_dt": round(self.training_dataset.trajs[0].logged_dt / self.training_dataset.trajs[0].lammps_dt),
                
        #         # TODO: load folder should follow somehow model from WANDB
        #         "load_folder": self.load_folder,
                
        #         "loss_func": self.loss_func_name,
        #         "normalize_loss": self.normalize_loss,
        #         "optimizer": self.optimizer_name,
        #         "log_lr": self.log_lr,
        #         "log_weight": self.log_weight,
        #         "scheduler": self.scheduler_name,
        #         "scheduling_factor": self.scheduling_factor,
        #         "scheduling_freq": self.scheduling_freq,

        #         "eval_dataset_steps": self.eval_dataset_steps,
        #         "eval_steps_per_dt": self.eval_steps_per_dt,
        #         "eval_init_skip": self.eval_init_skip,
        #         })
