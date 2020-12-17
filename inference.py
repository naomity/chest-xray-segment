from data import load_data
from model import unet
from loss_metric import dice_coef, iou_coef
import keras
import argparse
import sys
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
# data loading args
parser.add_argument('--input', type=str, help='input npy file')
parser.add_argument('--input-shape', default=(224, 224, 3), help='input shape of image')
parser.add_argument('--normalize', action='store_true', help='normalize input or scale down to [0,1]')
# inference stage args
parser.add_argument('--pretrain-path', type=str, nargs='+', default=[None], help='path to pre-trained model weight')
parser.add_argument('--save_dir', type=str, default=None, help='directory to save predicted masks')
# parser.add_argument('--metric', type=str, nargs='+', default=['accuracy'], help='metrics to calculate')
# model config
parser.add_argument('--model', default='unet', help='select for segment model backbone')
parser.add_argument('--attention', action='store_true', help='if attention should be used during upsampling')
parser.add_argument('--dilation', action='store_true', help='if dilation convs should be used in bottlenecks')
parser.add_argument('--residual', action='store_true', help='if residual connections are desired')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.model == 'unet':
        model = unet(num_filters=32, kernel_size=3, input_shape=args.input_shape,
                     num_classes=args.num_classes, pretrain_path=args.pretrain_path,
                     attention=args.attention, residual=args.residual, dilation=args.dilation)
    else:
        print('only supporting unet for now')
        sys.exit()

    model.compile(optimizer=keras.optimizers.Adam(3e-4), metrics=[dice_coef, iou_coef],
                  loss='binary_crossentropy')

    img, mask, file_list = load_data(args.input, args.input_shape, args.normalize)

    print(model.evaluate(img, mask))

    pred = model.predict(img)

    if args.save_dir is not None:
        for i, fn in enumerate(file_list):
            plt.imsave(args.save_dir + fn, pred[i], cmap='gray')
