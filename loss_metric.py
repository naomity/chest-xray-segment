from keras import backend as K
import math
import tensorflow as tf
import numpy as np


def dice_coef(y_true, y_pred):
    # y_true_f = K.batch_flatten(y_true)
    # y_pred_f = K.batch_flatten(y_pred)
    # intersection = K.sum(y_true_f * y_pred_f, axis = -1)
    # return (2. * intersection + K.epsilon()) / (K.sum(y_true_f, axis=-1) + K.sum(y_pred_f,axis=-1) + K.epsilon())
    intersection = K.sum(y_true * y_pred, axis=[2,3])
    sums = K.sum(y_true, axis=[2,3]) + K.sum(y_pred, axis=[2,3])
    dice = (2. * intersection) / (sums)
    return K.mean(dice, axis=-1)


def iou_coef(y_true, y_pred):
    # y_true_f = K.batch_flatten(y_true)
    # y_pred_f = K.batch_flatten(y_pred)
    # intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    # union = K.sum(y_true_f,axis=-1) + K.sum(y_pred_f,axis=-1) - intersection
    # return (intersection + K.epsilon()) / (union + K.epsilon())
    intersection = K.sum(y_true * y_pred, axis=[2, 3])
    union = K.sum(y_true, axis=[2, 3]) + K.sum(y_pred, axis=[2, 3]) - intersection
    iou = intersection / union
    return K.mean(iou, axis=-1)


def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


# losses
def focal_loss(gamma=2., alpha=.5):
    def focal(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, K.epsilon(), 1. - K.epsilon())
        pt_0 = K.clip(pt_0, K.epsilon(), 1. - K.epsilon())

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal


def manha_loss(y_true, y_pred): # does not work
    return K.sum(K.abs(y_true - y_pred), axis=1, keepdims=True)


# def surface_loss(y_true, y_pred):
#     def calc_dist_map(seg):
#         res = np.zeros_like(seg)
#         posmask = seg.astype(np.bool)
#         if posmask.any():
#             negmask = ~posmask
#             res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
#         return res
#
#     def calc_dist_map_batch(y_true):
#         y_true_numpy = y_true.numpy()
#         return np.array([calc_dist_map(y) for y in y_true_numpy]).astype(np.float32)
#
#     y_true_dist_map = tf.py_function(func=calc_dist_map_batch, inp=[y_true], Tout=tf.float32)
#     multipled = y_pred * y_true_dist_map
#     return K.mean(multipled)


def tversky_loss(beta): # adds a weight beta to FP & FN, ==dice when beta = 0.5
  def loss(y_true, y_pred):
    numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
    return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)
  return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


def lovasz_hinge(y_true, y_pred, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """

    def lovasz_hinge_flat(logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
          ignore: label to ignore
        """

        def compute_loss():
            labelsf = tf.cast(labels, logits.dtype)
            signs = 2. * labelsf - 1.
            errors = 1. - logits * tf.stop_gradient(signs)
            errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
            gt_sorted = tf.gather(labelsf, perm)
            grad = lovasz_grad(gt_sorted)
            loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
            return loss

        # deal with the void prediction case (only void pixels)
        loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                       lambda: tf.reduce_sum(logits) * 0.,
                       compute_loss,
                       strict=True,
                       name="loss"
                       )
        return loss

    def flatten_binary_scores(scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = tf.reshape(scores, (-1,))
        labels = tf.reshape(labels, (-1,))
        if ignore is None:
            return scores, labels
        valid = tf.not_equal(labels, ignore)
        vscores = tf.boolean_mask(scores, valid, name='valid_scores')
        vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
        return vscores, vlabels

    logits = y_pred
    labels = y_true
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


# def Weighted_Hausdorff_loss(y_true, y_pred, batch_size=16):
#     # https://arxiv.org/pdf/1806.07564.pdf
#     #prob_map_b - y_pred
#     #gt_b - y_true
#
#     def tf_repeat(tensor, repeats):
#         with tf.compat.v1.variable_scope('repeat'):
#             expanded_tensor = np.expand_dims(tensor, -1)
#             multiples = [1] + repeats
#             tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
#             repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
#         return repeated_tensor
#
#     dim = 224
#     n_pixels = dim * dim
#     max_dist = math.sqrt(dim**2 + dim**2)
#     all_img_locations = tf.convert_to_tensor(cartesian([np.arange(dim), np.arange(dim)], tf.float32))
#
#     terms_1 = []
#     terms_2 = []
#     y_true = tf.squeeze(y_true, axis=-1)
#     y_pred = tf.squeeze(y_pred, axis=-1)
# #     y_true = tf.reduce_mean(y_true, axis=-1)
# #     y_pred = tf.reduce_mean(y_pred, axis=-1)
#     for b in range(batch_size):
#         gt_b = y_true[b]
#         prob_map_b = y_pred[b]
#         # Pairwise distances between all possible locations and the GTed locations
#         n_gt_pts = tf.reduce_sum(gt_b)
#         gt_b = tf.where(tf.cast(gt_b, tf.bool))
#         gt_b = tf.cast(gt_b, tf.float32)
#         d_matrix = tf.sqrt(tf.maximum(tf.reshape(tf.reduce_sum(gt_b*gt_b, axis=1), (-1, 1)) +
#                                       tf.reduce_sum(all_img_locations*all_img_locations, axis=1)-
#                                       2*(tf.matmul(gt_b, tf.transpose(all_img_locations))), 0.0))
#         d_matrix = tf.transpose(d_matrix)
#         # Reshape probability map as a long column vector,
#         # and prepare it for multiplication
#         p = tf.reshape(prob_map_b, (n_pixels, 1))
#         n_est_pts = tf.reduce_sum(p)
#         p_replicated = tf_repeat(tf.reshape(p, (-1, 1)), [1, n_gt_pts])
#         eps = 1e-6
#         alpha = 4
#         # Weighted Hausdorff Distance
#         term_1 = (1 / (n_est_pts + eps)) * tf.reduce_sum(p * tf.reshape(tf.reduce_min(d_matrix, axis=1), (-1, 1)))
#         d_div_p = tf.reduce_min((d_matrix + eps) / (p_replicated**alpha + eps / max_dist), axis=0)
#         d_div_p = tf.clip_by_value(d_div_p, 0, max_dist)
#         term_2 = tf.reduce_mean(d_div_p, axis=0)
#         terms_1.append(term_1)
#         terms_2.append(term_2)
#     terms_1 = tf.stack(terms_1)
#     terms_2 = tf.stack(terms_2)
#     terms_1 = tf.print(tf.reduce_mean(terms_1), [tf.reduce_mean(terms_1)], "term 1")
#     terms_2 = tf.print(tf.reduce_mean(terms_2), [tf.reduce_mean(terms_2)], "term 2")
#     res = terms_1 + terms_2
#     return res
