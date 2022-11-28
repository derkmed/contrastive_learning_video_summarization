import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from model import SummarizerModel
from dataset import SummarizerDataset
import params
import numpy as np
from tqdm import tqdm

def train_epoch(run_id, learning_rate, epoch, data_loader, model, criterion, criterion_metric, optimizer, writer, use_cuda=True):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
        writer.add_scalar('Learning Rate', learning_rate, epoch)

        print("Learning rate is: {}".format(param_group['lr']))

    losses, accuracy = [], []

    model.train()

    for i, data in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        inputs = torch.stack(data['segments'])
        inputs = inputs.permute(1,0,5,2,3,4).cuda()
        label = data['scores'].cuda()
        output = model(inputs)

        loss = criterion(output, label.flatten())
        acc = criterion_metric(output, label.flatten())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accuracy.append(acc)
        if i % 24 == 0:
            print(
                f'- Training Epoch {epoch}\tLoss: {np.mean(losses) :.5f}\tAccuracy: {np.mean(accuracy) :.5f}')

    print('Training Epoch: %d\tLoss: %.4f\tAccuracy: %.4f' % (epoch,  np.mean(losses), np.mean(accuracy)))
    writer.add_scalar('Training Loss', np.mean(losses), epoch)
    writer.add_scalar('Training Accuracy', np.mean(accuracy), epoch)

    del loss, inputs, output, label

    return model, np.mean(losses)


def train(run_id, **kwargs):
    writer = SummaryWriter(os.path.join(os.curdir + '/logs', str(run_id)))
    save_dir = os.path.join(os.path.join(os.curdir + '/models', str(run_id)))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset = SummarizerDataset(
        'summe', data_list_file='/home/hojaeyoon/summe/train_test_split.txt', gt_root='/home/hojaeyoon/summe/GT')
    train_dataloader = DataLoader(dataset, batch_size=1)
    model = SummarizerModel(freeze_tclr=True).cuda()
    criterion = nn.CrossEntropyLoss()
    criterion_metric = nn.MSELoss()

    for epoch in range(params.num_epochs):
        print(f'Epoch {epoch}')
        start = time.time()
        
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        model, train_loss = train_epoch(
            run_id, params.learning_rate, epoch, train_dataloader, model, criterion, criterion_metric, optimizer, writer)
        time_per_epoch = time.time()-start
        accuracy = criterion_metric()
        print(f'Time: {time_per_epoch}\tLoss: {train_loss.item()}\tAccuracy: {}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TCLR + Transformer train script')
    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default="dummy_linear",
                        help='run_id')
    parser.add_argument("--restart", action='store_true', required=False, default=False)
    parser.add_argument("--saved_model", dest='saved_model',
                        type=str, required=False, default=None, help='run_id')
    args = parser.parse_args()
    train(args.run_id, **args)
