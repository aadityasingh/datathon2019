import pdb

# Torch imports
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable

# Numpy & Scipy imports
import numpy as np
import scipy
import scipy.misc

from tensorboardX import SummaryWriter
import shutil
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Trainer:
    def __init__(self, model, train_loader, test_loader, params):
        """
        Args:
            model: 
            params: hyperparameters and other settings
        """
        self.model = model
        self.params = params
        self.params['start_epoch'] = 0

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.loss = nn.MSELoss()
        self.optimizer = self.get_optimizer()

        self.summary_writer = SummaryWriter(log_dir=self.params['summary_dir'])

    def train(self, opts):
        self.model.train()
        last_loss = float('inf')
        save_new_checkpoint = False
        for epoch in range(self.params['start_epoch'], self.params['num_epochs']):
            loss_list = []
            print("epoch {}...".format(epoch))
            for batch_idx, row in enumerate(tqdm(self.train_loader)):
                data, target = row[:, :-1], row[:, -1]
                if torch.cuda.is_available():
                    # print('using GPU')
                    data = data.cuda()
                data = Variable(data)

                self.optimizer.zero_grad()
                pred = self.model.forward(data)
                loss = self.loss(pred, target)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.data[0].item())

            print("epoch {}: - training loss: {}".format(epoch, np.mean(loss_list)))
            new_lr = self.adjust_learning_rate(epoch)
            print('learning rate:', new_lr)

            if epoch % (opts.test_every//2) == 0:
                new_loss = self.test(epoch)
                if new_loss < last_loss:
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    })
                    print("Saved new checkpoint!")
                last_loss = new_loss

            if epoch % opts.test_every == 0:
                self.summary_writer.add_scalar('training/loss', np.mean(loss_list), epoch)
                self.summary_writer.add_scalar('training/learning_rate', new_lr, epoch)
                # self.save_checkpoint({
                #     'epoch': epoch + 1,
                #     'state_dict': self.model.state_dict(),
                #     'optimizer': self.optimizer.state_dict(),
                # })
                # self.print_image("training/epoch"+str(epoch))

    def test(self, cur_epoch):
        print('testing...')
        self.model.eval()
        test_loss = 0
        mse_loss = 0
        kld_loss = 0
        for i, row in enumerate(self.test_loader):
            data, target = row[:, :-1], row[:, -1]
            if torch.cuda.is_available():
                data = data.cuda()
            data = Variable(data)

            pred = self.model.forward(data)
            test_loss += self.loss(pred, target).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        self.summary_writer.add_scalar('testing/loss', test_loss, cur_epoch)
        self.model.train()
        return test_loss

    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'], weight_decay=self.params['weight_decay']) 

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
        learning_rate = self.params['learning_rate'] * (self.params['learning_rate_decay'] ** (epoch//self.params['learning_rate_step']))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        return learning_rate

    def save_checkpoint(self, state, is_best=False, filename='checkpoint2.pth.tar'):
        '''
        a function to save checkpoint of the training
        :param state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        :param is_best: boolean to save the checkpoint aside if it has the best score so far
        :param filename: the name of the saved file
        '''
        torch.save(state, self.params['checkpoint_dir'] + filename)
        # if is_best:
        #     shutil.copyfile(self.args.checkpoint_dir + filename,
        #                     self.args.checkpoint_dir + 'model_best.pth.tar')
