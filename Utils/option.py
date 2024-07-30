import os
import warnings
import argparse


warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser()
# Directories
model_save_dir = os.path.join('results/', 'LOL-v1/', 'LOL-v1/')
parser.add_argument('--model_save_dir', type=str, default=model_save_dir)
parser.add_argument('--trainset_name', type=str, default='v1')
parser.add_argument('--testset_name', type=str, default='v1')
parser.add_argument('--trainset_path', type=str, default="your train folder/")
parser.add_argument('--testset_path', type=str, default="your test folder/")
# parser.add_argument('--trainset_len', type=int, default=485)

# Train
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--valid_epoch', type=int, default=1)
parser.add_argument('--lr_sche', default=True)
parser.add_argument('--no_lr_sche', type=str, default=False, help='no lr cos schedule')
parser.add_argument('--gps', type=int, default=3, help='residual_groups')
parser.add_argument('--blocks', type=int, default=2, help='residual_blocks')
parser.add_argument('--ema', type=str, default=True, help='use ema')
parser.add_argument('--momentum', type=float, default=0.999, help='ema decay rate')

parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--device', type=str, default='cuda:1', help='use gpu')
parser.add_argument('--seed', type=int, default=2021, help='seed')

# Test
parser.add_argument('--test_model', type=str, default='test model)')
parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])
parser.add_argument('--output_dir', type=str, default='save image folder')

args = parser.parse_args()
