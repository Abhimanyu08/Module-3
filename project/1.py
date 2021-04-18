import minitorch

a = minitorch.Parameter(minitorch.rand((2,3), backend= minitorch.make_tensor_backend(minitorch.FastOps)))
b = minitorch.Parameter(minitorch.rand((3,4), backend= minitorch.make_tensor_backend(minitorch.FastOps)))


c = a.value@b.value

print(c)