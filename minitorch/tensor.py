"""
Implementation of the core Tensor object for autodifferentiation.
"""

from .autodiff import FunctionBase, Variable
from . import operators
import random
from .tensor_ops import TensorOps
from .util import assert_close
from .tensor_data import TensorData


# Construction
def zeros(shape):
    return Tensor.make([0] * int(operators.prod(shape)), shape)


def rand(shape):
    """
    Produce a random tensor of size `shape`.

    Args:
       shape (tuple): shape of tensor.

    Returns:
       :class:`Tensor` : New tensor
    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    return Tensor.make(vals, shape)


def tensor(ls, shape=None):
    if not shape:
        shape = (len(ls),)
    return Tensor.make(ls, shape)


def ensure_tensor(b):
    if isinstance(b, (int, float)):
        return tensor([b])
    return b


# Tensor class
class Tensor(Variable):
    def __init__(self, v, back=None, name=None, backend=None):
        assert isinstance(v, TensorData)
        super().__init__(back, name=name)
        self._tensor = v
        self.tf = backend
        if backend is None:
            self.tf = TensorFunctions

    def _new(self, tensor_data):
        return Tensor(tensor_data, backend=self.tf)

    @staticmethod
    def make(storage, shape, strides=None, backend=None):
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def cuda(self):
        return Tensor(
            self._tensor, back=self.back, name=self.name, backend=CudaTensorFunctions
        )

    # Properties
    @property
    def shape(self):
        return self._tensor.shape

    @property
    def size(self):
        return self._tensor.size

    @property
    def dims(self):
        return self._tensor.dims

    def contiguous(self):
        return self.tf.Copy.apply(self)

    # Functions
    def __add__(self, b):
        return self.tf.Add.apply(self, ensure_tensor(b))

    def __sub__(self, b):
        return self.tf.Add.apply(self, -ensure_tensor(b))

    def __mul__(self, b):
        return self.tf.Mul.apply(self, ensure_tensor(b))

    def __truediv__(self, b):
        return self.tf.Mul.apply(self, tensor([1 / b]))

    def __lt__(self, b):
        return self.tf.LT.apply(self, ensure_tensor(b))

    def __gt__(self, b):
        return self.tf.LT.apply(ensure_tensor(b), self)

    def __neg__(self):
        return self.tf.Neg.apply(self)

    def sigmoid(self):
        return self.tf.Sigmoid.apply(self)

    def relu(self):
        return self.tf.ReLU.apply(self)

    def log(self):
        return self.tf.Log.apply(self)

    def sum(self, dim=None):
        return self.tf.Sum.apply(self, dim)

    def mean(self, dim=None):
        return self.tf.Mean.apply(self, dim)

    def permute(self, *order):
        return self.tf.Permute.apply(self, order)

    def view(self, *shape):
        return self.tf.View.apply(self, shape)

    def __repr__(self):
        return self._tensor.to_string()

    def __getitem__(self, key):
        return self._tensor.get(key)

    def __setitem__(self, key, val):
        self._tensor.set(key, val)

    @property
    def grad(self):
        return self.derivative

    def expand(self, other):
        ""
        if self.shape == other.shape:
            return other

        shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = zeros(shape)
        self.tf.id_map(other, out=buf)
        if self.shape == shape:
            return buf

        buf2 = zeros(self.shape)
        self.tf.add_reduce(buf, out=buf2)
        return buf2

    # Internal
    def zeros(self, shape=None):

        if shape is None:
            out = zeros(self.shape)
        else:
            out = zeros(shape)
        out.tf = self.tf
        return out

    def tuple(self):
        return self._tensor.tuple()

    # Extra
    def get_data(self):
        return Tensor(self._tensor)

    def backward(self, grad_output=None):
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = tensor([1.0])
            grad_output.tf = self.tf
        super().backward(grad_output)


# Constructors
class Function(FunctionBase):
    data_type = Tensor

    @staticmethod
    def variable(data, back):
        return Tensor(data[0], back, backend=data[1])

    @staticmethod
    def data(a):
        return (a._tensor, a.tf)


def make_tensor_functions(backend):
    class TF:
        neg_map = backend.map(operators.neg)
        sigmoid_map = backend.map(operators.sigmoid)
        relu_map = backend.map(operators.relu)
        log_map = backend.map(operators.log)
        id_map = backend.map(operators.id)

        add_zip = backend.zip(operators.add)
        mul_zip = backend.zip(operators.mul)
        lt_zip = backend.zip(operators.lt)
        relu_back_zip = backend.zip(operators.relu_back)
        log_back_zip = backend.zip(operators.log_back)

        add_reduce = backend.reduce(operators.add)

        class Neg(Function):
            @staticmethod
            def forward(ctx, t1):
                return TF.neg_map(t1)

            @staticmethod
            def backward(ctx, grad_output):
                return TF.neg_map(grad_output)

        class Add(Function):
            @staticmethod
            def forward(ctx, t1, t2):
                return TF.add_zip(t1, t2)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, grad_output

        class Mul(Function):
            @staticmethod
            def forward(ctx, a, b):
                raise NotImplementedError('Need to include this file from past assignment.')

            @staticmethod
            def backward(ctx, grad_output):
                raise NotImplementedError('Need to include this file from past assignment.')

        class Sigmoid(Function):
            @staticmethod
            def forward(ctx, a):
                raise NotImplementedError('Need to include this file from past assignment.')

            @staticmethod
            def backward(ctx, grad_output):
                raise NotImplementedError('Need to include this file from past assignment.')

        class ReLU(Function):
            @staticmethod
            def forward(ctx, a):
                raise NotImplementedError('Need to include this file from past assignment.')

            @staticmethod
            def backward(ctx, grad_output):
                raise NotImplementedError('Need to include this file from past assignment.')

        class Log(Function):
            @staticmethod
            def forward(ctx, a):
                raise NotImplementedError('Need to include this file from past assignment.')

            @staticmethod
            def backward(ctx, grad_output):
                raise NotImplementedError('Need to include this file from past assignment.')

        class Sum(Function):
            @staticmethod
            def forward(ctx, a, dim):
                ctx.save_for_backward(a.shape)
                if dim is not None:
                    return TF.add_reduce(a, [dim])
                else:
                    return TF.add_reduce(a, list(range(a.dims))).view(1)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        class Mean(Function):
            @staticmethod
            def forward(ctx, a, dim):
                raise NotImplementedError('Need to include this file from past assignment.')

            @staticmethod
            def backward(ctx, grad_output):
                raise NotImplementedError('Need to include this file from past assignment.')

        class LT(Function):
            @staticmethod
            def forward(ctx, a, b):
                raise NotImplementedError('Need to include this file from past assignment.')

            @staticmethod
            def backward(ctx, grad_output):
                raise NotImplementedError('Need to include this file from past assignment.')

        class Permute(Function):
            @staticmethod
            def forward(ctx, a, order):
                ctx.save_for_backward(order)
                return a._new(a._tensor.permute(*order))

            @staticmethod
            def backward(ctx, grad_output):
                order = ctx.saved_values
                order = [a[0] for a in sorted(enumerate(order), key=lambda a: a[1])]
                return grad_output._new(grad_output._tensor.permute(*order))

        class View(Function):
            @staticmethod
            def forward(ctx, a, shape):
                ctx.save_for_backward(a.shape)
                assert a._tensor.is_contiguous, "Must be contiguous to view"
                t = Tensor.make(a._tensor._storage, shape)
                t.tf = a.tf
                return t

            @staticmethod
            def backward(ctx, grad_output):
                original = ctx.saved_values
                ret = Tensor.make(grad_output._tensor._storage, original)
                ret.tf = grad_output.tf
                return ret

        class Copy(Function):
            @staticmethod
            def forward(ctx, a):
                return TF.id_map(a)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

    return TF


TensorFunctions = make_tensor_functions(TensorOps)

# Uncomment for Module 3
CudaTensorFunctions = None

# from cuda_ops import CudaOps
# CudaTensorFunctions = make_tensor_functions(CudaOps)


def central_difference(f, *vals, arg=0, epsilon=1e-6, ind=None):
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f, *vals):
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = central_difference(f, *vals, arg=i, ind=ind)
        assert_close(x.grad[ind], check)
