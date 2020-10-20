from torch.nn.modules.module import Module
from ..functions.reflection_pad3d import ReflectionPad3dFunction

class ReflectionPad3d(Module):

    def __init__(self, pad):
        super(ReflectionPad3d, self).__init__()

        self.pad = pad

    def forward(self, input):
        return ReflectionPad3dFunction.apply(input, self.pad)

