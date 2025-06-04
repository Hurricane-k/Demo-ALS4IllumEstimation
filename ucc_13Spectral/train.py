#%% python package
import torch
import os
import logging
from src.ucc13 import UCC, custom_loss
from utils.uccDataset import uccDataset
from utils.metrics import angular_error_torch
from utils.augmentation import Compose, Resize
from torch import optim, utils
from config.settings import DEVICE, reproduce
from config.args import parse_func
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

#%% organzie pytorch to python-lightning
class LightningTrainer(L.LightningModule):

    def __init__(
            self, 
            model, 
            loss, 
            args, 
            device = DEVICE, # remove .cuda() / .to(DEVICE)
            use_spec = True,
            data_transform=None, 
            save_dir=None):
        super().__init__()
        self.model = model()
        self.loss_func = loss
        # pytorch-lightning can remove .cuda() / .to(device)
        self.model = self.model.to(device)
        self.data_transform = data_transform
        self.use_spec = use_spec
        self.best_ae = float('inf')
        self.validation_aes = []
        self.save_dir = save_dir
        self.best_ae_records = []
        self.best_epoch = 0
        self.args = args # parse_arg

    def training_step(self, batch, batch_idx):
        self.model.train()
        if self.data_transform is not None:
            batch = self.data_transform(batch)
        # pytorch-lightning can remove .cuda() / .to(device)
        inputs_x1 = batch['input']
        inputs_x2 = batch['spect']
        targets = batch['target']

        if self.use_spec:
            out = self.model(inputs_x1,inputs_x2)
        else:
            out = self.model(inputs_x1,None)
        loss = self.loss_func(out, targets)

        return loss
    
    def on_train_epoch_end(self):
        self.lr_schedulers().step()

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        inputs_x1 = batch['input']
        inputs_x2 = batch['spect']
        illum = batch['illum']

        with torch.no_grad():
            if self.use_spec:
                out = self.model.inference(inputs_x1, inputs_x2)
            else:
                out = self.model.inference(inputs_x1, None)
        try:
            ae = angular_error_torch(out, illum)
        except:
            print(out)
            exit(0)
        self.validation_aes.append(ae)

        return ae
    
    def on_validation_epoch_end(self):
        aes = torch.stack(self.validation_aes)
        avg_ae = aes.mean()

        if avg_ae < self.best_ae and self.current_epoch > 3:
            self.best_ae = avg_ae
            self.best_epoch = self.current_epoch
            torch.save(aes, os.path.join(self.save_dir, 'best_ae_records.pth'))
            self.trainer.save_checkpoint(
                os.path.join(self.save_dir, 'best.ckpt'))
        self.validation_aes = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs_x1 = batch['input']
        inputs_x2 = batch['spect']
        if self.use_spec:
            y_hat = self.model(inputs_x1,inputs_x2)
        else:
            y_hat = self.model(inputs_x1,None)
        return y_hat

    def configure_optimizers(self):
        if self.args.optimizer == "adam":
            optimizer_class = torch.optim.Adam
            optimizer_args = {'lr': self.args.lr, 'betas': (0.9, 0.999)}
        elif self.args.optimizer == "adamw":
            optimizer_class = torch.optim.AdamW
            optimizer_args = {
                'lr': self.args.lr,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-2
            }
        else:
            raise (NotImplementedError(
                "Optimizer {} is not support now, please choose from {}".
                format(self.args.optimizer, ['adam', 'adamw'])))

        model_params = self.parameters()

        self.optimizer = optimizer_class(model_params, **optimizer_args)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            total_iters=self.trainer.max_epochs,
            start_factor=1,
            end_factor=0.1)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, self.trainer.max_epochs, eta_min=1e-4)
        return [self.optimizer], [self.scheduler]
    
#%% main function
if __name__ == '__main__':
    # initialization and parse_args
    torch.set_float32_matmul_precision('high')
    args = parse_func()
    model_name = args.model_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    fold = args.fold_num
    if args.use_spec == 1:
        use_spec = True 
    else:
        use_spec = False
    reproduce(seed=args.random_seed)

    checkpoint_path = args.checkpoint_path
    data_dir = args.data_dir
    val_dir = data_dir
    num_workers = args.num_workers
    
    dataset_name = (data_dir.split('/')[-1] 
                    if data_dir.split('/')[-1] != '' 
                    else data_dir.split('/')[-2])
    if use_spec:
        project_name = dataset_name+'_'+model_name+'_'+'wSpectral'
    else:
        project_name = dataset_name+'_'+model_name+'_'+'woSpectral'
    if use_spec:
        save_dir = os.path.join(checkpoint_path, 
                                project_name, 
                                'fold_'+str(fold))
    else:
        save_dir = os.path.join(checkpoint_path, 
                                project_name, 
                                'fold_'+str(fold))
    os.makedirs(save_dir, exist_ok=True)

    # no data augment
    train_transform = Compose([
        # RandomCrop(size=0.7, ratio=0.8),
        Resize((args.input_size, args.input_size)),
    ])

    val_transform = Compose([
        Resize((args.input_size, args.input_size))])

    print('===============\n')
    logging.info(f'''
        Starting training:
        Project Name:          {project_name}
        Model Name:            {args.model_name}
        Use Spectral info:     {use_spec}
        Train Dataset:         {data_dir}
        Fold Num:              {fold}
        Epochs:                {epochs}
        Batch size:            {batch_size}
        Learning rate:         {lr}
        Augmentation:          {train_transform}
        save_dictionary:       {save_dir}
    ''')
    print('===============')

    # define model, dataloader, loss function
    model = UCC 
    dataloader=uccDataset
    loss_function=custom_loss

    traniner = LightningTrainer(
        model=model,
        args=args,
        loss=loss_function,
        use_spec = use_spec,
        save_dir=save_dir)
    
    train_dataset = dataloader(
        data_dir,
        mode='train',
        fold = fold,
        transform=train_transform)

    valset = dataloader(
        val_dir,
        mode='test', 
        # in train.py, except mode = 'train', all others are considered as validation 
        transform=val_transform,
        fold = fold)

    train_loader = utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True)

    val_loader = utils.data.DataLoader(
        valset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename='{epoch:02d}',
        every_n_epochs=args.save_iter,
        save_top_k=-1)
    
    trainer = L.Trainer(
        max_epochs=epochs,
        precision="32",
        accelerator='gpu',
        enable_checkpointing=True,
        gradient_clip_val=.1,
        callbacks=[checkpoint_callback])

    trainer.fit(
        model=traniner,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader)

    print('===============')
    print(
        "Best average angular error: {}, found at epoch: {}".format(
            traniner.best_ae, 
            traniner.best_epoch))
    print('===============')
    print("Model saved to: {}".format(save_dir))
    print('===============')

