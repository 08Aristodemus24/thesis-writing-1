from keras import backend as K



def dice_coef(y_true, y_pred, smooth=100):
    """
    according to twnsorflow the formula for calculating the
    Dice-Sorensen Coefficient (DSC) is as follows:

    DSC = (2 * sum(y_true * y_pred)) / (sum(y_true) + sum(y_pred))
    """

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return dice

def dice_sorensen_coefficient_loss(y_true, y_pred):
    """
    the full DSC loss is calculated by taking the resulting
    DSC value and subtracting 1 by it.
    """

    DSC = dice_coef(y_true, y_pred)
    loss = 1 - DSC

    return loss

# usage of this is as follows:
# model.compile(
#     optimizer=Adam(learning_rate=rec_alpha),
#     loss=dice_sorensen_coefficient_loss,
#     metrics=protocols[protocol]['metrics']
# )