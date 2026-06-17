# coding=utf-8
from __future__ import (absolute_import, division, unicode_literals)

import os
import sys
import time
import torch
import logging
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import optim, distributed
# from torch.cuda.amp import GradScaler
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import csv
from params import get_args, save_hp_to_json
from modules import CLIP4Clip, convert_weights
from modules import SimpleTokenizer as ClipTokenizer
from modules.file import PYTORCH_PRETRAINED_BERT_CACHE
from dataloaders.data_dataloaders import DATALOADER_DICT
from utils.lr_scheduler import lr_scheduler
from utils.optimization import BertAdam, prep_optim_params_groups
from utils.log import setup_primary_logging, setup_worker_logging
from utils.misc import set_random_seed, convert_models_to_fp32, save_checkpoint
from utils.dist_utils import is_master, get_rank, is_dist_avail_and_initialized, init_distributed_mode
from utils.metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
from modules.clip import force_flush_debug_buffer
best_R1 = 0
 

DEBUG_REPLAY_ONLY = True

def save_last_checkpoint(model, optimizer, scaler, epoch, global_step,
                         best_R1, model_dir, filename='ckpt.pth.tar'):
    """
    항상 마지막 학습 상태를 저장하는 checkpoint
    eval 전에 무조건 저장하는 용도
    기존 save_checkpoint 스타일과 동일하게 동작
    """

    state = {
        'epoch': epoch + 1,
        'global_step': global_step,
        'arch': 'CLIp4Clip',
        'state_dict': model.state_dict(),
        'best_acc1': best_R1,
        'optimizer': optimizer.state_dict(),
    }

    if scaler is not None:
        state['scaler'] = scaler.state_dict()

    filepath = os.path.join(model_dir, filename)

    torch.save(state, filepath)

    logging.info(f"[checkpoint] Saved LAST checkpoint to {filepath}")

def get_debug_model(model):
    return model.module if hasattr(model, "module") else model

def get_traj_debug_stats(model):

    m = get_debug_model(model)

    stats = {}

    try:
        visual = m.clip.visual

        if hasattr(visual, "last_prompt_stats"):
            stats.update(visual.last_prompt_stats)

        tp = visual.TemporalPrompt

        if hasattr(tp, "last_traj_stats"):
            stats.update(tp.last_traj_stats)

    except Exception as e:
        stats["debug_error"] = str(e)

    return stats

def summarize_direction_logits(logits, labels, num_classes):

    with torch.no_grad():

        probs = torch.softmax(logits.detach().float(), dim=-1)

        preds = torch.argmax(logits, dim=-1)

        pred_counts = torch.bincount(
            preds.cpu(),
            minlength=num_classes
        ).float()

        pred_ratio = (
            pred_counts /
            pred_counts.sum().clamp(min=1)
        )

        entropy = -(
            probs * torch.log(probs + 1e-12)
        ).sum(dim=-1).mean().item()

        return {
            "entropy": entropy,
            "pred_ratio": pred_ratio.tolist(),
            "avg_logits":
                logits.detach().float().mean(dim=0).cpu().tolist(),
            "batch_acc":
                (preds == labels).float().mean().item(),
        }

def main(args):
    """main function"""

    set_random_seed(args.seed)

    # Set multiprocessing type to spawn.
    if not args.remote:
        torch.multiprocessing.set_start_method('spawn')

    # Set logger
    log_queue = setup_primary_logging(os.path.join(args.output_dir, "log.txt"), args.log_level, args.remote)

    # lmdb
    if args.lmdb_dataset not in [None, 'None']:
        assert os.path.exists(args.lmdb_dataset)
        print('INFO: [dataset] Using {} as data source'.format(args.lmdb_dataset))

    # the number of gpus
    args.ngpus_per_node = torch.cuda.device_count()
    print("INFO: [CUDA] The number of GPUs in this node is {}".format(args.ngpus_per_node))

    # Distributed training = training on more than one GPU.
    # Also easily possible to extend to multiple nodes & multiple GPUs.
    args.distributed = (args.gpu is None) and torch.cuda.is_available() and (not args.dp)
    if args.distributed:
        if args.remote:
            raise NotImplementedError
        else:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, log_queue, args))
    else:
        # nn.DataParallel (DP)
        if args.dp:
            args.gpu, args.world_size = args.multigpu[0], len(args.multigpu)
        else:
            args.world_size = 1
        main_worker(args.gpu, None, log_queue, args)

def main_worker(gpu, ngpus_per_node, log_queue, args):
    """main worker"""
    global best_R1
    args.gpu = gpu

    ## ####################################
    # initilization
    ## ####################################
    global_rank = init_distributed_mode(args, ngpus_per_node, gpu)
    setup_worker_logging(global_rank, log_queue, args.log_level)
    # Lock the random seed of the model to ensure that the model initialization of each process is the same.
    set_random_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # save parameters
    if is_master(): save_hp_to_json(args.output_dir, args)

    ## ####################################
    # create model
    ## ####################################
    # create tokenizer
    tokenizer = ClipTokenizer()
    model_state_dict = torch.load(args.init_model, map_location='cpu') if args.init_model else None
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    
    model = CLIP4Clip.from_pretrained(args.cross_model, 
                                        cache_dir=cache_dir,
                                        state_dict=model_state_dict,
                                        task_config=args)
    if is_master():
        logging.info(f"[traj_prompt] use_traj_prompt = {args.use_traj_prompt}")
        logging.info(f"[traj_prompt] traj_prompt_scale = {args.traj_prompt_scale}")
        logging.info(f"[traj_prompt] traj_feat_type = {args.traj_feat_type}")
    #model.freeze_cip_layers(args.freeze_layer_num)
    
    #logging.info('\nweight from DeepCluster')
     

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision in ["amp", "fp32"]or args.gpu is None:  	# or args.precision == "fp32" 
        logging.info("[weight convert] ==>> Convert weights to fp32 for {}...".format(args.precision))
        convert_models_to_fp32(model)
        logging.info("[weight convert] ==>> Convert done!")

    if not torch.cuda.is_available():
        model.float()
        logging.warning("using CPU, this will be slow")
    else:
        model.cuda(args.gpu)
        if args.precision == "fp16":
            convert_weights(model)
        # Previously batch size and workers were global and not per GPU.
        # args.batch_size = args.batch_size / ngpus_per_node)
        # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.distributed:
            if args.freeze_clip:
               print('do freeze_clip:',args.freeze_clip)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                                find_unused_parameters=True if args.freeze_clip else True)
        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", get_rank() % ngpus_per_node)

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT
    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
    num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                    / args.gradient_accumulation_steps) * args.epochs

    ## ####################################
    # optimization strategies
    ## ####################################
    optimizer_grouped_parameters = prep_optim_params_groups(args, model, coef_lr=args.coef_lr)
    scaler = GradScaler() if args.precision == "amp" else None
    if args.optim == 'BertAdam':
        logging.info('[optimizer] Using BertAdam Optimizer...')
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                                schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                                t_total=num_train_optimization_steps, weight_decay=args.wd,
                                max_grad_norm=1.0)
        scheduler = None
    elif args.optim == 'AdamW':
        logging.info('[optimizer] Using AdamW Optimizer...')
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr,
                                betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.wd)
        scheduler = lr_scheduler(mode='cos', init_lr=args.lr, all_iters=num_train_optimization_steps,
                                    slow_start_iters=args.warmup_proportion * num_train_optimization_steps,
                                    weight_decay=args.wd
                                )
    else:
        raise NotImplementedError

    if is_master():
        tf_writer = SummaryWriter(args.tensorboard_path)
    else:
        tf_writer = None

    ## ####################################
    #  optionally resume from a checkpoint
    ## ####################################
    start_epoch, global_step = 0, 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)

            if not args.load_from_pretrained:
                if "optimizer" in checkpoint and optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if "scaler" in checkpoint and scaler is not None:
                    logging.info("[resume] => Loading state_dict of AMP loss scaler")
                    scaler.load_state_dict(checkpoint['scaler'])
                start_epoch, global_step = checkpoint["epoch"], checkpoint["global_step"]

            logging.info(f"[resume] => loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})\n")
        else:
            logging.info("[resume] => no checkpoint found at '{}'\n".format(args.resume))

    ## ####################################
    # train and evalution
    ## ####################################
    if is_master():
        logging.info("\n======================== Running training ========================")
        logging.info("  Num examples = %d", train_length)
        logging.info("  Batch size = %d", args.batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)
        logging.info("\n======================== Running test ========================")
        logging.info("  Num examples = %d", test_length)
        logging.info("  Batch size = %d", args.batch_size_val)
        logging.info("  Num steps = %d", len(test_dataloader))
        logging.info("\n======================== Running val ========================")
        logging.info("  Num examples = %d", val_length)

    all_start = time.time()

    if args.do_eval and is_master():
        R1, infer_epoch_time, info_str = eval_epoch(model, test_dataloader, device, args=args)
        torch.cuda.synchronize()
        all_time = time.time() - all_start
        logging.info('The total running time of the program is {:.2f} Seconds\n'.format(all_time))
        logging.info('The maximum GPU memory occupied by this program is {:.2f} GB\n'.format(
                        torch.cuda.max_memory_allocated(0) * 1.0 / 1024 / 1024 / 1024))
        sys.exit(0)

    eval_infer_times = []
    best_e = 0
    best_info = []
    for epoch in range(start_epoch, args.epochs):
        if is_dist_avail_and_initialized():
            train_sampler.set_epoch(epoch)
        # set_random_seed(epoch + args.seed)

        tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, optimizer,
                                            global_step, scaler=scaler, tf_writer=tf_writer, scheduler=scheduler)

        if is_master():
            logging.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
            
            #CKPT 저장
            save_last_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_R1=best_R1,
                model_dir=args.output_dir,
                filename="ckpt.pth.tar"
            )
            logging.info("[checkpoint] Saved last checkpoint before evaluation.")
            #요기까지
            
            # Run on val dataset, this process is *TIME-consuming*.
            R1, infer_epoch_time, info_str = eval_epoch(model, test_dataloader, device, args=args, epoch=epoch)
            eval_infer_times.append(infer_epoch_time)
            if best_R1 <= R1:
                best_R1 = R1
                best_e = epoch
                best_info = info_str
            logging.info("The best R1 is: {:.4f}, best_e={}\n".format(best_R1, best_e))
            # save checkpoint
            ckpt_dict = {
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': 'CLIp4Clip',
                    'state_dict': model.state_dict(),
                    'best_acc1': best_R1,
                    'optimizer': optimizer.state_dict(),
                }
            if scaler is not None: ckpt_dict['scaler'] = scaler.state_dict()
            save_checkpoint(ckpt_dict, best_R1 <= R1, args.output_dir, filename='ckpt.pth.tar')

    all_time = time.time() - all_start

    if is_master():
        logging.info('The total running time of the program is {:.1f} Hour {:.1f} Minute\n'.format(all_time // 3600, 
                    all_time % 3600 / 60))
        logging.info('The average inference time of {} runs is {:.2f} Seconds\n'.format(args.epochs, np.mean(eval_infer_times)))
        logging.info('The maximum GPU memory occupied by this program is {:.2f} GB\n'.format(
                    torch.cuda.max_memory_allocated(0) * 1.0 / 1024 / 1024 / 1024))
        logging.info("The best R1 is: {:.4f}, best_epoch={}\n".format(best_R1, best_e))
        for info in best_info:
            logging.info(info)
        print("The above program id is {}\n".format(args.output_dir))

    torch.cuda.empty_cache()
    sys.exit(0)

def train_epoch(epoch, args, model, train_dataloader, device, optimizer, global_step,
                scheduler=None, scaler=None, tf_writer=None):
    samples_per_epoch = len(train_dataloader.dataset)

    # torch.cuda.empty_cache()
    model.train()

    if epoch == 0 and is_master():
        no_clip = args.new_added_modules
        trainable_size =0
        total_param_size  = 0  
        for name, param in model.module.named_parameters():
            if param.requires_grad==True:
                total_param_size += param.numel() 
                trainable_size += param.numel() 
                param_size_MB = param.numel()/(1000**2)
                # logging.info(f'trainerble parameters are: {name}, size is {param_size_MB:.4f} MB')
            else:
                total_param_size += param.numel() 
        trainable_size_MB = trainable_size/(1000**2)
        total_param_size_MB = total_param_size/(1000**2)
        percentage = (trainable_size / total_param_size)*100
        # logging.info("Trainable param percentage are: {}".format(percentage))
        # logging.info("Trainable params are: {} MB, Total params are: {} MB".format(trainable_size_MB,total_param_size_MB))

    total_loss = 0

    end = time.time()
    if args.datatype == "direction_mcq":
        loss_name = "ClsLoss"
    else:
        loss_name = "SimLoss"
    for step, batch in enumerate(train_dataloader):
        # if step == 1:
        #     break
        # logging.info(f"Step: {step}   Batch: {batch}")
        
        optimizer.zero_grad()
        if scheduler is not None: scheduler(optimizer, global_step=global_step)
        # multi-gpu does scattering it-self
#여기서부터
        # if torch.cuda.is_available():
        #     batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        # input_ids, input_mask, segment_ids, video, video_mask = batch
        # data_time = time.time() - end
        
        # # logging.info(f"Step: {step}   input_ids: {input_ids}")
        # # logging.info(f"Step: {step}   input_mask: {input_mask}")
        # # logging.info(f"Step: {step}   segment_ids: {segment_ids}")
        # # logging.info(f"Step: {step}   video: {video}")
        # # logging.info(f"Step: {step}   video_mask: {video_mask}")

        # # forward
        # # with torch.cuda.amp.autocast(enabled=scaler is not None):
        # with torch.amp.autocast('cuda', enabled=scaler is not None):
        #     output = model(input_ids, segment_ids, input_mask, video, video_mask)
        #     loss = output['loss'].mean()
        #     #cluster_loss = output['cluster_loss'].mean()
        #     sim_loss = output['sim_loss'].mean()
#여기까지

        # 새로 추가: 문자열 metadata 분리
        if args.datatype == "direction_mcq":
            (
                input_ids,
                input_mask,
                segment_ids,
                video,
                video_mask,
                labels,
                sample_ids,
                qa_types,
                answer_texts,
                questions,
                debug_video_paths,
            ) = batch

            if torch.cuda.is_available():
                input_ids = input_ids.to(device=device, non_blocking=True)
                input_mask = input_mask.to(device=device, non_blocking=True)
                segment_ids = segment_ids.to(device=device, non_blocking=True)
                video = video.to(device=device, non_blocking=True)
                video_mask = video_mask.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True).long()

            if is_master() and global_step % 50 == 0:
                label_count = torch.bincount(labels.detach().cpu(), minlength=args.num_classes)
                logging.info(
                    "[batch_label_dist] "
                    f"epoch={epoch} "
                    f"step={step} "
                    f"global_step={global_step} "
                    f"counts={label_count.tolist()}"
                )

            # class text는 모든 sample마다 동일하므로 첫 번째 것만 사용
            input_ids = input_ids[0]          # [9, max_words]
            input_mask = input_mask[0]        # [9, max_words]
            segment_ids = segment_ids[0]      # [9, max_words]

            # video는 [B, 1, T, C, H, W] 형태 그대로 둬도 기존 코드와 맞을 가능성이 큼
            if step == 0 and is_master():
                print("[DEBUG] input_ids:", input_ids.shape)
                print("[DEBUG] video:", video.shape)
                print("[DEBUG] video_mask:", video_mask.shape)
                print("[DEBUG] labels:", labels.shape)
            # 만약 shape 에러가 나면 그때 video.squeeze(1)을 시도
            # video = video.squeeze(1)
            # video_mask = video_mask.squeeze(1)
            data_time = time.time() - end

            with torch.amp.autocast('cuda', enabled=scaler is not None):
                output = (
                    model.module.forward_direction(
                        input_ids,
                        segment_ids,
                        input_mask,
                        video,
                        video_mask,
                        labels
                    )
                    if hasattr(model, "module")
                    else model.forward_direction(
                        input_ids,
                        segment_ids,
                        input_mask,
                        video,
                        video_mask,
                        labels
                    )
                )

                logits = output["logits"]      # [B, 9]
                if (
                    args.datatype == "direction_mcq"
                    and is_master()
                    and global_step % 100 == 0
                ):

                    log_stats = summarize_direction_logits(
                        logits,
                        labels,
                        args.num_classes
                    )

                    traj_stats = get_traj_debug_stats(model)

                    logging.info(
                        "[collapse_debug] "
                        f"epoch={epoch} "
                        f"step={step} "
                        f"global_step={global_step} "

                        f"batch_acc={log_stats['batch_acc']:.4f} "
                        f"entropy={log_stats['entropy']:.4f} "

                        f"pred_ratio={log_stats['pred_ratio']} "

                        f"avg_logits={log_stats['avg_logits']} "

                        f"traj_stats={traj_stats}"
                    )

                loss = output["loss"]
                sim_loss = output["sim_loss"]

                if logits.shape[0] != labels.shape[0] or logits.shape[1] != args.num_classes:
                    raise RuntimeError(
                        f"Bad logits shape: logits={logits.shape}, "
                        f"labels={labels.shape}, num_classes={args.num_classes}"
                    )

        else:
            input_ids, input_mask, segment_ids, video, video_mask, video_ids, video_paths = batch
            
            if torch.cuda.is_available():
                input_ids = input_ids.to(device=device, non_blocking=True)
                input_mask = input_mask.to(device=device, non_blocking=True)
                segment_ids = segment_ids.to(device=device, non_blocking=True)
                video = video.to(device=device, non_blocking=True)
                video_mask = video_mask.to(device=device, non_blocking=True)

            data_time = time.time() - end

            with torch.amp.autocast('cuda', enabled=scaler is not None):
                output = model(input_ids, segment_ids, input_mask, video, video_mask)
                #추가 디버깅
                if torch.isnan(output["loss"]).any() or torch.isinf(output["loss"]).any():
                    print("[DEBUG] loss broken")
                if torch.isnan(output["sim_loss"]).any() or torch.isinf(output["sim_loss"]).any():
                    print("[DEBUG] sim_loss broken")
                #여기까지
                loss = output['loss'].mean()
                sim_loss = output['sim_loss'].mean()

        # NaN 첫 발생 디버깅
        if torch.isnan(loss) or torch.isnan(sim_loss) or torch.isinf(loss) or torch.isinf(sim_loss):
            print("\n[DEBUG] NaN/Inf detected!")
            print(f"[DEBUG] epoch={epoch}, step={step}, global_step={global_step}")
            print(f"[DEBUG] video shape={video.shape}")
            print(f"[DEBUG] video_mask sum per sample={video_mask.sum(dim=-1)}")

            print(f"[DEBUG] input video has nan: {torch.isnan(video).any().item()}")
            print(f"[DEBUG] input video has inf: {torch.isinf(video).any().item()}")

            flat_video = video.view(video.size(0), -1)
            print(f"[DEBUG] per-sample video min: {flat_video.min(dim=1).values}")
            print(f"[DEBUG] per-sample video max: {flat_video.max(dim=1).values}")
            print(f"[DEBUG] per-sample video mean: {flat_video.mean(dim=1)}")

            force_flush_debug_buffer(
                header=f"[DEBUG] force flush because loss/sim_loss is NaN/Inf | epoch={epoch}, step={step}, global_step={global_step}"
            )

            raise RuntimeError("NaN detected in training")
        #여기까지

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
            
        # logging.info(f"Step: {step}   Loss: {loss}")
        # logging.info(f"Step: {step}   Sim_Loss: {sim_loss}")

        # update weights
        if scaler is not None:
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        if hasattr(model, 'module'):
            torch.clamp_(model.module.clip.logit_scale.data, 0.1, 4.6052)
        else:
            torch.clamp_(model.clip.logit_scale.data, 0.1, 4.6052)

        batch_time = time.time() - end
        end = time.time()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            global_step += 1
            if global_step % args.n_display == 0 and is_master():
                if args.datatype == "direction_mcq":
                    current_batch_size = labels.size(0)
                else:
                    current_batch_size = len(input_ids)

                num_samples = (step + 1) * current_batch_size * args.world_size
                percent_complete = num_samples * 1.0 / samples_per_epoch * 100
                logit_scale_data = model.module.clip.logit_scale.data if hasattr(model, 'module') \
                                    else model.clip.logit_scale.data
                lr_tmp = optimizer.param_groups[0]['lr'] if args.optim == 'AdamW' else \
                            optimizer.get_lr()[0]
                
                logging.info(
                    f"Epoch: {epoch} [{num_samples} ({percent_complete:.1f}%)]\t"
                    f"{loss_name}: {sim_loss.item():.4f} \t"
                    #f"SimLoss: {sim_loss.item():.4f} CLoss {cluster_loss.item():.4f}\t"
                    f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                    f"\tLR: {lr_tmp:.1e}\tlogit_scale {logit_scale_data:.3f}"
                )
                # tensorboard log
                log_data = {
                    "sim_loss": sim_loss.item(),
                    #"cluster_loss": cluster_loss.item(),
                    "data_time": data_time,
                    "batch_time": batch_time,
                    "scale": logit_scale_data.item(),
                    "lr": lr_tmp
                }
                for name, val in log_data.items():
                    name = "train/" + name
                    if tf_writer is not None:
                        tf_writer.add_scalar(name, val, global_step=global_step)

        total_loss += float(loss)

    total_loss = total_loss / len(train_dataloader)

    return total_loss, global_step

def eval_epoch_direction(model, test_dataloader, device, args=None, epoch=0):
    model.eval()

    total = 0
    correct = 0
    infer_start_t = time.time()

    all_preds = []
    all_labels = []
    all_qa_types = []
    all_sample_ids = []
    all_answer_texts = []
    all_questions = []
    all_video_paths = []

    with torch.no_grad():
        for bid, batch in enumerate(test_dataloader):
            (
                input_ids,
                input_mask,
                segment_ids,
                video,
                video_mask,
                labels,
                sample_ids,
                qa_types,
                answer_texts,
                questions,
                debug_video_paths,
            ) = batch

            if torch.cuda.is_available():
                input_ids = input_ids.to(device=device, non_blocking=True)
                input_mask = input_mask.to(device=device, non_blocking=True)
                segment_ids = segment_ids.to(device=device, non_blocking=True)
                video = video.to(device=device, non_blocking=True)
                video_mask = video_mask.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True).long()


            input_ids = input_ids[0]
            input_mask = input_mask[0]
            segment_ids = segment_ids[0]

            output = (
                model.module.forward_direction(
                    input_ids,
                    segment_ids,
                    input_mask,
                    video,
                    video_mask,
                    labels=None
                )
                if hasattr(model, "module")
                else model.forward_direction(
                    input_ids,
                    segment_ids,
                    input_mask,
                    video,
                    video_mask,
                    labels=None
                )
            )

            logits = output["logits"]
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            all_qa_types.extend(list(qa_types))
            all_sample_ids.extend(list(sample_ids))
            all_answer_texts.extend(list(answer_texts))
            all_questions.extend(list(questions))
            all_video_paths.extend(list(debug_video_paths))

            if (bid + 1) % args.n_display == 0 or (bid + 1) == len(test_dataloader):
                logging.info("{}/{}\r".format(bid + 1, len(test_dataloader)))

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    all_infer_time = time.time() - infer_start_t
    acc = correct * 100.0 / total

    info_str = []
    info_str.append("Direction Action Recognition:")
    info_str.append(" (metric) >>> ACC@1: {:.2f}".format(acc))
    if args.output_dir is not None:
        csv_path = os.path.join(args.output_dir, f"eval_predictions_epoch{epoch}.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_id",
                "qa_type",
                "question",
                "video_path",
                "pred_label",
                "gold_label",
                "gold_answer_text",
                "correct",
            ])

            for sid, qa, q, vp, pred, gold, ans in zip(
                all_sample_ids,
                all_qa_types,
                all_questions,
                all_video_paths,
                all_preds,
                all_labels,
                all_answer_texts,
            ):
                writer.writerow([
                    sid,
                    qa,
                    q,
                    vp,
                    pred,
                    gold,
                    ans,
                    int(pred == gold),
                ])

        logging.info(f"[eval] Saved prediction CSV to {csv_path}")

    # qa_type별 accuracy
    qa_type_stats = {}
    for pred, label, qa_type in zip(all_preds, all_labels, all_qa_types):
        if qa_type not in qa_type_stats:
            qa_type_stats[qa_type] = [0, 0]
        qa_type_stats[qa_type][1] += 1
        if pred == label:
            qa_type_stats[qa_type][0] += 1

    for qa_type, (c, n) in qa_type_stats.items():
        qa_acc = c * 100.0 / n
        info_str.append(f" {qa_type} ACC@1: {qa_acc:.2f} ({c}/{n})")

    for info in info_str:
        logging.info(info)

    return acc, all_infer_time, info_str

def eval_epoch(model, test_dataloader, device, args=None, epoch=0):
    """evaluation"""
    if args.datatype == "direction_mcq":
        return eval_epoch_direction(model, test_dataloader, device, args=args, epoch=epoch)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_ and is_master():
        logging.info("Eval under the multi-sentence per video clip setting.")
        logging.info("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        infer_start_t = time.time()
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch
            if args.save_feature_path is not None and os.path.exists(args.save_feature_path):
                if bid < 2000:
                    if args.datatype == 'msrvtt':
                        print('{}\t'.format(bid + 1), test_dataloader.dataset.data['video_id'].values[bid], end='\n')
                    if args.datatype == 'lsmdc':
                        if 'Harry_Potter' in test_dataloader.dataset.iter2video_pairs_dict[bid][0]:
                            print('{}\t'.format(bid + 1), test_dataloader.dataset.iter2video_pairs_dict[bid], end='\n')

            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                sequence_output = model(input_ids, segment_ids, input_mask)['sequence_output']
                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    visual_output = model(video=video, video_mask=video_mask)['visual_output']
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                output = model(input_ids, segment_ids, input_mask, video, video_mask)
                batch_sequence_output_list.append(output['sequence_output'])
                batch_list_t.append((input_mask, segment_ids,))

                batch_visual_output_list.append(output['visual_output'])
                batch_list_v.append((video_mask,))

            if (bid + 1) % args.n_display == 0 or ( bid + 1) == len(test_dataloader):
                logging.info("{}/{}\r".format(bid, len(test_dataloader)))

        if torch.cuda.is_available(): torch.cuda.synchronize()
        all_infer_time = time.time() - infer_start_t
        logging.info('The total model inference time of the program is {:.2f} Seconds\n'.format(all_infer_time))
        if args.inference_speed_test:
            return 0

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, 
                                            batch_sequence_output_list, batch_visual_output_list, args=args)

    if multi_sentence_:
        logging.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logging.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))

    else:
        logging.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logging.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    # return for final logging
    info_str = []
    info_str.append("Text-to-Video:")
    info_str.append(' (metric) >>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    info_str.append("Video-to-Text:")
    info_str.append(' (metric) >>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    for info in info_str: logging.info(info)
    R1 = tv_metrics['R1']

    return R1, all_infer_time, info_str

def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list, args=None):
    """"calculate the similarity between visual output and text output"""
    if hasattr(model, 'module'):
        model = model.module
    else:
        model = model

    sim_matrix = []
    
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    # if args.camoe_dsl:
    # 	print('Apply DSL')
    # 	# https://github.com/starmemda/CAMoE
    # 	sim_matrix_ = torch.from_numpy(sim_matrix)
    # 	sim_matrix = sim_matrix_ * F.softmax(sim_matrix_, dim=0) * len(sim_matrix_)
    # 	# sim_matrix = sim_matrix_ * F.softmax(sim_matrix_, dim=1) * len(sim_matrix_)
    # 	sim_matrix = sim_matrix.cpu().numpy()

    return sim_matrix

if __name__ == "__main__":
    args = get_args()

    main(args)