from numba import cuda
from minitorch.tensor_data import count

@cuda.jit(device=True)
def times(a, b):
   return a * b

# Main cuda launcher
@cuda.jit()
def my_func(inp, out_shape):
    # Create some local memory
    local = cuda.local.array(len(out_shape), dtype = int)

    # Find my position.
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    # Compute some information
    count(x, out_shape, local)

    inp = None
    # Compute some global value
    

def call(inp, out_shape):
    return my_func[1,32](inp,out_shape)

call(2, (2,3,2))