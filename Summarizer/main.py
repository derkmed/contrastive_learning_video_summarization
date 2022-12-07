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


    # WARNING (derekahmed): You most certainly will have to freeze the base. This network is way too large for our standards.
    model = TCLRSummarizer(args.pretrain_tclr_path, d_model=args.tclr_dim, freeze_base=args.freeze_base,
        heads=args.enc_head, enc_layers=args.enc_layers, dropout=args.dropout)
    model.cuda()
    print(f"Loaded model from {args.pretrain_tclr_path}")
    # TODO (derekahmed)
    # This does not work with 1 lonely Nvidia T4.
    # if args.distributed:
    #     if args.gpu is not None:
    #         torch.cuda.set_device(args.gpu)
    #         model.cuda(args.gpu)
    #         args.batch_size = int(args.batch_size / ngpus_per_node)
    #         args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
    #         args.num_thread_reader = int(args.num_thread_reader / ngpus_per_node)
    #         model = torch.nn.parallel.DistributedDataParallel(
    #             model, device_ids=[args.gpu]
    #         )
    #     else:
    #         model.cuda()
    #         model = torch.nn.parallel.DistributedDataParallel(model)
    # elif args.gpu is not None:
    #     torch.cuda.set_device(args.gpu)
    #     model = model.cuda(args.gpu)
    # else:
    #     model = torch.nn.DataParallel(model).cuda()
    print("Finished loading model")

    train_dataset = Summarizer_Dataset(
        data_list_file = args.training_dataset,
        req_segment_count=args.req_segment_count,
        num_frames_per_segment=args.num_frames_per_segment)
    test_dataset = Summarizer_Dataset(
        data_list_file = args.testing_dataset, 
        req_segment_count=args.req_segment_count,
        num_frames_per_segment=args.num_frames_per_segment)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = None
        test_sampler = None

    # (derekahmed): This was a hack to get around the fact that a single Nvidia T4 can only handle a
    # batch size of 2 :(. We upper bound the batch sizes and aggregate gradients accordingly.
    CS6998_05_BATCH_UBOUND = 2
    train_batch_size = args.batch_size
    use_train_batch_hack = False
    if args.batch_size > CS6998_05_BATCH_UBOUND:
        train_batch_size = CS6998_05_BATCH_UBOUND
        use_train_batch_hack = True
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
        shuffle=(train_sampler is None), drop_last=True, num_workers=args.num_thread_reader,
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
        evaluate(test_loader, model, epoch, tb_logger, criterion_c, args)
        return

    # Configure optimizer.
    base_params = []
    importance_params = []
    for name, param in model.named_parameters():
        if "base" in name and "fc" not in name:
            base_params.append(param)
        else:
            importance_params.append(param)
    if args.optimizer == "adam":
        # optimizer = torch.optim.Adam(model.parameters(), args.lrv, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam([
                {"params": base_params, "lr": args.lr_base},
                {"params": importance_params, "lr": args.lr_importance},
            ], args.lrv, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD([
                {"params": base_params, "lr": args.lr_base},
                {"params": importance_params, "lr": args.lr_importance},
            ], args.lrv, momentum=args.momemtum,
            weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)
    
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
    total_batch_size = args.world_size * train_batch_size
    log("Starting training loop for rank: {}, total batch size: {}".format(args.rank, total_batch_size), args)
    log("Training dataset has {} videos".format(len(train_dataset)), args)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if (epoch + 1) % 5 == 0:
          evaluate(test_loader, model, epoch, tb_logger, criterion_c, args)
        
        train(train_loader, model, criterion_c, optimizer, scheduler, epoch, train_dataset, use_train_batch_hack, args)
        if args.rank == 0 and (epoch + 1) % args.checkpoint_cadence == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                checkpoint_dir,
                epoch + 1
            )


def train(train_loader, model, criterion, optimizer, scheduler, epoch, dataset, use_train_batch_hack, args):
    if use_train_batch_hack:
        log(f"Using Batch Training Hack to accumulate batches of size {args.batch_size}" +\
            f" on {len(dataset)} samples", args)
        TrainHack(train_loader, model, criterion, optimizer, scheduler, epoch, dataset, args)
    else:
        log("Using regular Batch Training.", args)
        losses = []
        s = time.time()
        for i_batch, sample_batch in enumerate(train_loader):
            batch_loss = TrainOneBatch(model, optimizer, scheduler, sample_batch, criterion, args)
            losses.append(batch_loss)
            if args.verbose and args.rank == 0:
                d = time.time() - s
                current_lr = optimizer.param_groups[0]["lr"]
                log("Epoch %d, Elapsed Time: %.3f, Epoch status: %.4f, Batch loss: %.6f, Learning rate: %.6f"
                    % (epoch + 1, d, args.batch_size * args.world_size * float(i_batch) / len(dataset),
                        batch_loss, current_lr), args)
        if args.rank == 0:
            d = time.time() - s
            current_lr = optimizer.param_groups[0]["lr"]
            log("Epoch %d, Elapsed Time: %.3f, Training loss: %.6f, Learning rate: %.6f"
                % (epoch + 1, d, np.mean(losses), current_lr), args)


def TrainOneBatch(model, opt, scheduler, data, loss_fun, args):
    segments = data["segments"].float().cuda(args.gpu, non_blocking=args.pin_memory)
    label_scores = data["label_scores"].cuda(args.gpu, non_blocking=args.pin_memory).view(-1)
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

def TrainHack(train_loader, model, criterion, optimizer, scheduler, epoch, dataset, args):
    '''
    Attempts to emulate larger batch sizes in the absence of GPU resources.
    Refer to https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually -to-zero-in-pytorch/4903/20
    '''

    def _TrainOneBatchHack(model, data, loss_fun, args):
        segments = data["segments"].float().cuda(args.gpu, non_blocking=args.pin_memory)
        label_scores = (data["label_scores"].cuda(args.gpu, non_blocking=args.pin_memory).view(-1))
        with torch.set_grad_enabled(True):
            video_embed, score = model(segments)
            if args.distributed:
                label_scores = allgather(label_scores, args)
                video_embed = allgather(video_embed, args)
                score = allgather(score, args)
            loss = loss_fun(score.view(-1), label_scores)

        gradient = torch.ones((loss.shape[0]), dtype=torch.long)\
            .cuda(args.gpu, non_blocking=args.pin_memory)
        loss.backward(gradient=gradient)
        loss = loss.sum()
        return loss.item()

    
    losses = []
    s = time.time()    
    batch_aggregation_cadence = int(args.batch_size / train_loader.batch_size)
    batch_loss = 0
    n_samples = 0
    optimizer.zero_grad()
    for i_batch, sample_batch in enumerate(train_loader):
        batch_loss = batch_loss + _TrainOneBatchHack(model, sample_batch, criterion, args)
        n_samples = n_samples + sample_batch['segments'].shape[0]
        if (i_batch + 1) % batch_aggregation_cadence == 0 or i_batch == len(train_loader) - 1:
            print(f"Aggregating {n_samples} samples on batch={i_batch + 1}.")
            losses.append(batch_loss)
            batch_loss = 0
            n_samples = 0
            optimizer.step()
            scheduler.step()
            if args.verbose and args.rank == 0:
                d = time.time() - s
                current_lr = optimizer.param_groups[0]["lr"]
                log("Epoch %d, Elapsed Time: %.3f, Epoch status: %.4f, Batch loss: %.6f, Learning rate: %.6f"
                    % (epoch + 1, d, args.batch_size * args.world_size * float(i_batch) / len(dataset),
                        batch_loss, current_lr), args)
    
    if args.rank == 0:
        d = time.time() - s
        current_lr = optimizer.param_groups[0]["lr"]
        log("Epoch %d, Elapsed Time: %.3f, Training loss: %.6f, Learning rate: %.6f"
            % (epoch + 1, d, np.mean(losses), current_lr), args)


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

    # (derekahmed) Collected the summary sums for debugging.
    debug_machine_sums = []
    debug_gt_sums = []
    with torch.no_grad():
        for i_batch, data in enumerate(test_loader):
            segments = data["segments"].float().cuda()
            labels = data["labels"].cuda().view(-1)
            label_scores = data["label_scores"].cuda().view(-1)
            video_embed, score = model(segments)
            # TODO(derekahmed) #debugging
            # print(f"Scores: ", score.shape)
            # print(f"Scores: ", score[0,:30].flatten())

            if args.distributed:
                labels = allgather(labels, args)
                label_scores = allgather(label_scores, args)
                video_embed = allgather(video_embed, args)
                score = allgather(score, args)

            if args.rank == 0:
                loss = loss_fun(score.view(-1), label_scores)
                n_segments = len(labels)
                summary_ids = (
                    score.detach().cpu().view(-1).topk(int(args.k * n_segments))[1]
                )
                summary = np.zeros(n_segments)
                summary[summary_ids] = 1
                # NOTE: Given the nature of this algorithm, summary will be an array
                # of repeated values based on args.k!
                f_score, precision, recall, debugging_info = evaluate_summary(
                    summary, 
                    labels.detach().cpu().numpy()
                    # label_scores.detach().cpu().numpy()
                )
                overlap_duration, machine_sum, gt_sum = debugging_info
                debug_machine_sums.append(machine_sum)
                debug_gt_sums.append(gt_sum)

                # TODO(derekahmed) #debugging delete me
                # print(f"Overlap duration: " + str(overlap_duration))
                loss = loss.mean()
                losses.update(loss.item(), video_embed.shape[0])
                f_scores.update(f_score, video_embed.shape[0])
                precisions.update(precision, video_embed.shape[0])
                recalls.update(recall, video_embed.shape[0])

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
            "Loss {loss.val:.6f} ({loss.avg:.6f})\t".format(
                epoch, f_score, precision, recall, loss=losses
            ),
            args,
        )
        table.add_row([f_score, precision, recall, loss])
        tqdm.write(str(table))

        # log("Machine Summary Sum: " + str(debug_machine_sums), args)
        # log("GT Summary Sum: " + str(debug_gt_sums), args)

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


def save_checkpoint(state, checkpoint_dir, epoch, n_ckpt=5):
    torch.save(
        state, os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch))
    )
    if epoch >= n_ckpt:
        oldest_ckpt = os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch - n_ckpt))
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

if __name__ == "__main__":
    main()
