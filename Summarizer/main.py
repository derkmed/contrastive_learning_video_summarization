import os
import random
from tabnanny import check
import time
import glob
import sys
from collections import OrderedDict
from prettytable import PrettyTable
from tqdm import tqdm
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from args import get_args
from summarizer_dataset import Summarizer_Dataset
from model import TCLRSummarizer
from metrics import compute_metrics
from utils import (
    AllGather,
    get_cosine_schedule_with_warmup,
    setup_for_distributed,
    Logger,
    evaluate_summary,
    AverageMeter,
)

allgather = AllGather.apply


def main():
    args = get_args()
    if args.verbose:
        print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Starting with random seed {args.seed}")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        args.world_size = 1
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.distributed.barrier()
        setup_for_distributed(args.rank == 0)


    model = TCLRSummarizer(args.pretrain_tclr_path)
    print("Created model")

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
            args.num_thread_reader = int(args.num_thread_reader / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    print("Finished loading model")

    train_dataset = Summarizer_Dataset(
        data_list_file = "../data/splits/augmented_tvsum_80.txt",
        req_segment_count=args.req_segment_count,
        num_frames_per_segment=args.num_frames_per_segment)
    test_dataset = Summarizer_Dataset(
        data_list_file = "../data/splits/20p_tvsum_list.txt", 
        req_segment_count=args.req_segment_count,
        num_frames_per_segment=args.num_frames_per_segment)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), drop_last=False, num_workers=args.num_thread_reader,
        pin_memory=args.pin_memory, sampler=train_sampler,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_val,
        shuffle=False, drop_last=False, num_workers=args.num_thread_reader,
        sampler=test_sampler,
    )

    # start a logger
    args.log_name = "{}_model_bs_{}_lr_{}_nsegments_{}_nfps_{}_nheads_{}_nenc_{}_dropout_{}".format(
        args.log_name, args.batch_size, args.lrv, args.req_segment_count, args.num_frames_per_segment,
        args.heads, args.enc_layers, args.dropout)
    tb_logdir = os.path.join(args.log_root, args.log_name)
    tb_logger = Logger(tb_logdir)
    if args.rank == 0:
        os.makedirs(tb_logdir, exist_ok=True)

    criterion_c = nn.MSELoss(reduction="none")
    
    # Evaluation loop.
    if args.evaluate:
        print("starting eval...")
        # TODO (derekahmed) fix me
        evaluate(test_loader, model, epoch, tb_logger, criterion_c, args)
        return

    # Configure optimizer.
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lrv, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lrv, momentum=args.momemtum,
            weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=1.0)
    
    checkpoint_dir = os.path.join(os.path.dirname(__file__), args.checkpoint_dir, args.log_name)
    if args.checkpoint_dir != "" and args.rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    # Optionally resume from a checkpoint.
    if args.resume:
        checkpoint_path = get_last_checkpoint(checkpoint_dir)
        if checkpoint_path:
            log("=> loading checkpoint '{}'".format(checkpoint_path), args)
            checkpoint = torch.load(checkpoint_path)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            log(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    checkpoint_path, checkpoint["epoch"]
                ),
                args,
            )
        else:
            log("=> no checkpoint found at '{}'".format(args.resume), args)

    if args.cudnn_benchmark:
        cudnn.benchmark = True
    total_batch_size = args.world_size * args.batch_size
    log("Starting training loop for rank: {}, total batch size: {}".format(args.rank, total_batch_size), args)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if (epoch + 1) % 2 == 0:
            evaluate(test_loader, model, epoch, tb_logger, criterion_c, args)
        
        # train for one epoch
        train(train_loader, model, criterion_c, optimizer, scheduler, epoch,
            train_dataset, tb_logger, args)
        if args.rank == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                checkpoint_dir,
                epoch + 1,
            )


def train(train_loader, model, criterion, optimizer, scheduler, epoch, dataset, tb_logger, args):
    running_loss = 0.0
    s = time.time()
    for i_batch, sample_batch in enumerate(train_loader):
        batch_loss = TrainOneBatch(model, optimizer, scheduler, sample_batch, criterion, 
            args, epoch)
        running_loss += batch_loss
        if args.verbose and args.rank == 0:
            d = time.time() - s
            if args.finetune:
                current_lr = optimizer.param_groups[1]["lr"]
            else:
                current_lr = optimizer.param_groups[0]["lr"]
            log(
                "Epoch %d, Elapsed Time: %.3f, Epoch status: %.4f, Training loss: %.4f, Learning rate: %.6f"
                % (
                    epoch + 1,
                    d,
                    args.batch_size * args.world_size * float(i_batch) / len(dataset),
                    running_loss / args.log_freq,
                    current_lr,
                ),
                args,
            )
            # log training data into tensorboard
            if tb_logger is not None:
                logs = OrderedDict()
                logs["Train loss"] = running_loss / args.log_freq
                logs["Learning rate"] = current_lr
                # how many iterations we have trained
                iter_count = epoch * len(train_loader) + i_batch
                for key, value in logs.items():
                    tb_logger.log_scalar(value, key, iter_count)
                tb_logger.flush()

            s = time.time()
            running_loss = 0.0


def TrainOneBatch(model, opt, scheduler, data, loss_fun, args, epoch):
    segments = data["segments"].float().cuda(args.gpu, non_blocking=args.pin_memory)
    label_scores = (
        data["label_scores"].cuda(args.gpu, non_blocking=args.pin_memory).view(-1)
    )

    # video = video / 255.0 This no longer makes sense for TCLR
    opt.zero_grad()
    with torch.set_grad_enabled(True):
        video_embd, score = model(segments)
        if args.distributed:
            label_scores = allgather(label_scores, args)
            video_embd = allgather(video_embd, args)
            score = allgather(score, args)
        loss = loss_fun(score.view(-1), label_scores)

    gradient = torch.ones((loss.shape[0]), dtype=torch.long)\
        .cuda(args.gpu, non_blocking=args.pin_memory)
    loss.backward(gradient=gradient)
    loss = loss.mean()
    opt.step()
    scheduler.step()
    return loss.item()


def evaluate(test_loader, model, epoch, tb_logger, loss_fun, args):
    losses = AverageMeter()
    f_scores = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    model.eval()
    if args.rank == 0:
        table = PrettyTable()
        table.title = "Eval result of epoch {}".format(epoch)
        table.field_names = ["F-score", "Precision", "Recall", "Loss"]
        table.float_format = "1.3"

    with torch.no_grad():
        for i_batch, data in enumerate(test_loader):
            segments = data["segments"].float().cuda()
            label_scores = data["label_scores"].cuda().view(-1)
             # video = video / 255.0 This no longer makes sense for TCLR
            video_embd, score = model(segments)
            if args.distributed:
                label_scores = allgather(label_scores, args)
                video_embd = allgather(video_embd, args)
                score = allgather(score, args)

            if args.rank == 0:
                loss = loss_fun(score.view(-1), label_scores)
                # summary_frames = torch.argmax(score.data, 1)

                # score = nn.functional.log_softmax(score.view(-1).detach().cpu(), dim=0)
                # score = nn.functional.normalize(score.view(-1).detach().cpu(), dim=0)
                summary_ids = (
                    score.detach().cpu().view(-1).topk(int(0.50 * len(label_scores)))[1]
                )
                summary = np.zeros(len(label_scores))
                summary[summary_ids] = 1
                # threshold = 0.85 * summary_frames.max()
                # print("Thershold: ", threshold)
                # summary_frames[summary_frames < threshold] = 0
                # summary_frames[summary_frames > threshold] = 1

                # print(
                #     "Summary frames: ",
                #     summary,
                #     "Labels: ",
                #     label,
                #     "Scores: ",
                #     score.view(-1),
                #     "label_scores: ",
                #     label_scores,
                # )

                f_score, precision, recall = evaluate_summary(
                    summary, label_scores.detach().cpu().numpy()
                )
                loss = loss.mean()
                losses.update(loss.item(), video_embd.shape[0])
                f_scores.update(f_score, video_embd.shape[0])
                precisions.update(precision, video_embd.shape[0])
                recalls.update(recall, video_embd.shape[0])

    loss = losses.avg
    f_score = f_scores.avg
    precision = precisions.avg
    recall = recalls.avg

    if args.rank == 0:
        log(
            "Epoch {} \t"
            "F-Score {} \t"
            "Precision {} \t"
            "Recall {} \t"
            "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                epoch, f_score, precision, recall, loss=losses
            ),
            args,
        )
        table.add_row([f_score, precision, recall, loss])
        tqdm.write(str(table))

        if tb_logger is not None:
            # log training data into tensorboard
            logs = OrderedDict()
            logs["Val_IterLoss"] = losses.avg
            logs["F-Score"] = f_scores.avg
            logs["Precision"] = precisions.avg
            logs["Recall"] = recalls.avg

            # how many iterations we have validated
            for key, value in logs.items():
                tb_logger.log_scalar(value, key, epoch)

            tb_logger.flush()

    model.train()


def save_checkpoint(state, checkpoint_dir, epoch, n_ckpt=10):
    torch.save(
        state, os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch))
    )
    if epoch - n_ckpt >= 0:
        oldest_ckpt = os.path.join(
            checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch - n_ckpt)
        )
        if os.path.isfile(oldest_ckpt):
            os.remove(oldest_ckpt)


def get_last_checkpoint(checkpoint_dir):
    all_ckpt = glob.glob(os.path.join(checkpoint_dir, "epoch*.pth.tar"))
    if all_ckpt:
        all_ckpt = sorted(all_ckpt)
        return all_ckpt[-1]
    else:
        return ""


def log(output, args):
    print(output)
    with open(
        os.path.join(
            os.path.dirname(__file__), "output_log", args.log_name + ".txt"
        ),
        "a",
    ) as f:
        f.write(output + "\n")


# gcn_params = []
# base_params = []
# for name, param in model.named_parameters():
#     if 'ga' in name or 'gcn' in name:
#         gcn_params.append(param)
#     else:
#         base_params.append(param)

# optimizer = optim.Adam([
#     {"params": base_params, "lr": args.learning_rate_lstm},
#     {"params": gcn_params, "lr":args.learning_rate_gcn},
# ], weight_decay=1e-6)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.4)


if __name__ == "__main__":
    main()
