# python libraries
import os
import json
import time
import datetime
import argparse
import numpy as np
# pytorch libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.models import create_model
from dataset import build_dataset
from engine import train_one_epoch, evaluate
from models import *
from losses import LogitDistillationLoss, FeatureDistillationLoss, TotalDistillationLoss


def str_to_dict(string):
    return json.loads(string)


def get_args_parser():
    parser = argparse.ArgumentParser('HDKD training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=200, type=int)

    # Model parameters
    parser.add_argument('--model', default='teacher_model', type=str, metavar='MODEL',
                        help='Name of model to train whether the teacher model, student model or HDKD model') 
    parser.add_argument('--input-size', default=[224,224], type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')


    args, _ = parser.parse_known_args()
    assert args.model in ['HDKD', 'student_model', 'teacher_model']
    if args.model == 'HDKD':
        parser.add_argument('--teacher-path', type=str,
                            default='/content/91.43.pth')
        parser.add_argument('--_lambda', type=float,
                            default=10, help="The weight of the contribution of Feature distillation loss on the total loss")
        # Logit Distillation parameters
        parser.add_argument('--lkd_distillation-type', default='soft',
                            choices=['none', 'soft', 'hard'], type=str, help="")
        parser.add_argument('--lkd_distillation-alpha', default=0.5, type=float, help="This coefficient balances the KL loss term on the teacher prediction and cross-entropy (CE) loss term on the actual label")
        parser.add_argument('--lkd_distillation-tau', default=1.0, type=float, help="Temperature to control the softness of the probability distirbution")
        # Feature Distillation parameters
        parser.add_argument('--teacher_layers', nargs='+', type=int, default=[1, 2, 3], help='List of teacher layers indices to perform feature distillation')
        parser.add_argument('--student_layers', nargs='+', type=int, default=[1, 2, 3], help='List of student layers indices to perform feature distillation')
        parser.add_argument('--fkd_distillation_alpha', default=1.0, type=float,help='A weighting parameter that give weight to each layer in the total feature distillation loss, higher alpha will give much higher weights to later features')



    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')


    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=2e-5, metavar='LR',
                        help='warmup learning rate (default: 2e-5)')
    parser.add_argument('--min-lr', type=float, default=2e-5, metavar='LR', 
                        help='lower lr bound for cyclic schedulers that hit 0 (2e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters

    parser.add_argument('--rotate', type=int, default=180,
                        help='Rotate the image by the given degrees (default: 180)')
    parser.add_argument('--horizontal_flip', action='store_true', default=True,
                        help='Apply horizontal flip (default: true)')
    parser.add_argument('--vertical_flip', action='store_true', default=True,
                        help='Apply vertical flip (default: true)')
    parser.add_argument('--brightness', type=float, default=0.3,
                        help='Adjust the brightness. 1 means no change (default:0.3)')
    parser.add_argument('--contrast', type=float, default=0.3,
                        help='Adjust the contrast. 1 means no change (default:0.3)')
    parser.add_argument('--saturation', type=float, default=None,
                        help='Adjust the saturation. 1 means no change (default:None)')
    parser.add_argument('--hue', type=float, default=None,
                        help='Adjust the hue. 0 means no change (default:None)')
    parser.add_argument('--sharpness', type=float, default=2,
                        help='Adjust the sharpness. 1 means no change (default:2)')
    parser.add_argument('--blur', type=int, default=3,
                        help='Apply Gaussian blur with the given radius (default:3)')

    # Smote parameters
    parser.add_argument('--use_smote', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--data-path', default="HAM-10000/", type=str,
                        help='dataset path'),
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for saving in the same directory')
    parser.add_argument('--seed', default=0, type=int)


    return parser



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    start_time = time.time()

    if args.use_smote:
        with open("classes_distribution.json", "r") as file:
            args.classes_distribution = json.load(file)

    print("")
    train_loader, args.nb_classes = build_dataset(True, args)
    val_loader, _, = build_dataset(False, args)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        drop=args.drop
    )
    model.to(device)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The {} has number of params = {}'.format(args.model,num_parameters))

    teacher_model=None
    use_distillation=False
    if args.model=="HDKD":
        use_distillation=True
        teacher_model = create_model(
        "teacher_model",
        num_classes=args.nb_classes,
        drop=args.drop)
        teacher_model.load_state_dict(torch.load(args.teacher_path), strict=False)
        teacher_model.eval();
        teacher_model.to(device)

    if args.model=="teacher_model":
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        scheduler = None
    else:
        optimizer = create_optimizer(args,model)
        scheduler, _ = create_scheduler(args, optimizer)


    criterion={'CE_loss':nn.CrossEntropyLoss()}
    if use_distillation:
        lkd_criterion = LogitDistillationLoss(nn.CrossEntropyLoss(),teacher_model,args.lkd_distillation_type,args.lkd_distillation_alpha,args.lkd_distillation_tau)
        fkd_criterion = FeatureDistillationLoss(teacher_model, model, args.fkd_distillation_alpha, args.teacher_layers, args.student_layers)
        total_distillation = TotalDistillationLoss(lkd_criterion,fkd_criterion,args._lambda)
        criterion['Total_distillation_loss'] = total_distillation

    best_val_accuracy=0
    for epoch in range(args.epochs):

        train_stats = train_one_epoch(
                    model, criterion, train_loader,
                    optimizer, device, epoch, use_distillation = use_distillation
                )

        if scheduler is not None:
            scheduler.step(epoch)
            print(scheduler.optimizer.param_groups[0]['lr'])

        val_criterion=nn.CrossEntropyLoss()
        test_stats = evaluate(model, val_criterion, val_loader, device, use_distillation = use_distillation)

        print(f"Epoch [{epoch+1}/{args.epochs}], "  
          f"Training Loss: {train_stats['loss']:.4f}, Training Accuracy: {train_stats['accuracy']:.2%}, "
          f"Validation Loss: {test_stats['loss']:.4f}, Validation Accuracy: {test_stats['accuracy']:.2%}")


        val_accuracy = test_stats['accuracy']
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print("----------- saving --------------")
            torch.save(model.state_dict(), args.output_dir + 'best_model.pth')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__  == '__main__':
    args, unknown = get_args_parser().parse_known_args()
    main(args)