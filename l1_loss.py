import mxnet as mx

def get_l1_absolute_loss():
    origin = mx.symbol.Variable("origin")
    rec = mx.symbol.Variable("rec")
    diff = origin - rec
    absolute = mx.symbol.abs(data=diff)
    mean = mx.symbol.mean(data=absolute)
    return mx.symbol.MakeLoss(data=mean)

