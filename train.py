import argparse
import sys
from model import unet
from data import load_data, data_augment
from sklearn.model_selection import KFold, train_test_split
import keras
from loss_metric import dice_coef, iou_coef, dice_loss, focal_loss

parser = argparse.ArgumentParser()
# data loading args
parser.add_argument('--input', type=str, help='input npy file')
parser.add_argument('--val-input', type=str, default=None, help='validation input npy file')
parser.add_argument('--input-shape', default=(224, 224, 1), help='input shape of image')
parser.add_argument('--augment', default=True, help='augment input images')
parser.add_argument('--normalize', action='store_true', help='normalize input or scale down to [0,1]')
# train stage args
parser.add_argument('--validation-split', type=float, default=0.2, help='the split of validation data')
parser.add_argument('--cross-validation', type=int, default=0, help='number of cross validation folds')
parser.add_argument('--num-classes', type=int, default=1, help='number of classification classes')
parser.add_argument('--activation', type=str, default='sigmoid', help='number of classification classes')
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='training epochs')
parser.add_argument('--loss', default='binary_crossentropy', help='training loss function')
parser.add_argument('--weight-prefix', type=str, default='classify_model', help='name of the trained weight')
parser.add_argument('--pretrain-path', type=str, nargs='+', default=[None], help='path to pre-trained model weight')
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
    if args.loss == 'dice':
        loss = dice_loss
    elif args.loss == 'focal':
        loss = focal_loss(2., 0.5)
    else:
        loss = args.loss
    model.compile(optimizer=keras.optimizers.Adam(3e-4), metrics=[dice_coef, iou_coef],
                  loss=loss)
    if args.pretrain_path[0] is not None:
        model.load_weights(args.pretrain_path[0])

    img, mask, _ = load_data(args.input, args.input_shape, args.normalize)
    if args.val_input:
        x_val, y_val, _ = load_data(args.val_input, args.input_shape, args.normalize)
    else:
        x_val, y_val = None, None

    if args.cross_validation:
        kfold = KFold(n_splits=args.cross_validation, shuffle=True, random_state=42)
        count = 1
        for train, val in kfold.split(img, mask):
            x_train, x_val, y_train, y_val = img[train], img[val], mask[train], mask[val]
            if args.augment:
                x_train, y_train = data_augment(x_train, y_train)
            callbacks = [
                keras.callbacks.EarlyStopping(patience=10),
                keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=2, min_lr=1e-5, verbose=1),
                keras.callbacks.ModelCheckpoint(args.weight_prefix + str(count) + '.h5', verbose=1,
                                                save_best_only=True, save_weights_only=True)
            ]
            results1 = model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=2,
                                 validation_data=[x_val, y_val],
                                 callbacks=callbacks)
            count += 1
    else:
        if x_val is not None:
            x_train, y_train = img, mask
        else:
            x_train, x_val, y_train, y_val = train_test_split(img, mask, shuffle=True,
                                                              test_size=args.validation_split, random_state=42)
        if args.augment:
            x_train, y_train = data_augment(x_train, y_train)
        callbacks = [
            keras.callbacks.EarlyStopping(patience=20),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, verbose=1),
            keras.callbacks.ModelCheckpoint(args.weight_prefix + '.h5', verbose=1,
                                            save_best_only=True, save_weights_only=True)
        ]
        results1 = model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=2,
                             callbacks=callbacks,
                             validation_data=[x_val, y_val])
