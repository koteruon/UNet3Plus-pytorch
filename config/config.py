from yacs.config import CfgNode as CN

cfg = CN()

# MODEL
cfg.model = CN()
cfg.model.encoder = "resnet18"
cfg.model.pretrained = False
cfg.model.skip_ch = 64
cfg.model.use_cgm = False
cfg.model.aux_losses = 2
cfg.model.dropout = 0.3
cfg.model.n_tr = 0.0
cfg.model.alpha = 2.0

# DATA
cfg.data = CN()
cfg.data.type = "voc2012_aug"
cfg.data.data_dir = "./data"
cfg.data.crop_size = 512
cfg.data.num_classes = 21
cfg.data.batch_size = 1
cfg.data.num_workers = 0
cfg.data.max_training_samples = -1

# HYPERPARAMETERS
cfg.train = CN()
cfg.train.seed = 42
cfg.train.epochs = 10

cfg.train.lr = 0.001
cfg.train.lrf = 0.0005  # final lr
cfg.train.scheduler = "cyclic"
cfg.train.warmup_iters = 500

cfg.train.optimizer = "adamw"
cfg.train.weight_decay = 0.0001
cfg.train.momentum = 0.9
cfg.train.nesterov = True

cfg.train.accum_steps = 2
cfg.train.resume = ""
cfg.train.epochs = 52
cfg.train.val_interval = 1
cfg.train.device = "cuda"

cfg.train.aux_weight = 0.4
cfg.train.loss_type = "focal"
cfg.train.save_name = "UNet3Plus"

# LOGGING
cfg.train.logger = CN()
cfg.train.logger.log_dir = "./runs"

# tensorboard setting
cfg.train.logger.use_tensorboard = True
cfg.train.logger.tensorboard = CN()

# wandb setting
cfg.train.logger.use_wandb = False
cfg.train.logger.wandb = CN()
cfg.train.logger.wandb.project = "UNet3Plus"
cfg.train.logger.wandb.run_id = ""

# HYPERPARAMETERS
cfg.test = CN()
cfg.test.seed = 42
cfg.test.device = "cuda"
cfg.test.weight = ""
cfg.test.aux_weight = 0.4
cfg.test.loss_type = "focal"
cfg.test.save_name = "UNet3Plus"

# LOGGING
cfg.test.logger = CN()
cfg.test.logger.log_dir = "./test_runs"

# tensorboard setting
cfg.test.logger.use_tensorboard = True
cfg.test.logger.tensorboard = CN()

# wandb setting
cfg.test.logger.use_wandb = False
cfg.test.logger.wandb = CN()
cfg.test.logger.wandb.project = "UNet3Plus"
cfg.test.logger.wandb.run_id = ""
