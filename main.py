from eval import _eval
from train import _train
from Utils.option import *
from Utils.set_seed import *
from Utils.utils import *
import warnings
warnings.filterwarnings('ignore')


def main():
    if args.mode == 'train':
        trainLogger = create_log(args.mode)
        trainLogger.write('-' * 100)
        opt_k = vars(args)
        for k in opt_k.keys():
            trainLogger.write(f'\n---{k}: {opt_k[k]}')
        trainLogger.write('\n')
        trainLogger.write('-' * 100)
        trainLogger.write('\n')

        _train(trainLogger)

    elif args.mode == 'test':
        if not os.path.exists(args.output_dir) and args.mode == 'test':
            os.makedirs(args.output_dir)
        testLogger = create_log(args.mode)

        _eval(testLogger)


if __name__ == '__main__':
    seed_setting(args.seed)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.result_output_dir = args.output_dir
    main()
