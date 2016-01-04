from collections import deque
from numba import cuda

if not hasattr(cuda, 'event'):
    raise Exception('All CUDA examples in this directory use features in the '
                    'upcoming NumbaPro 0.11 which is scheduled for release '
                    'on the week of April 14.')

class MM(object):
    """Memory Manager
    
    Maintain a freelist of device memory for reuse.
    """
    def __init__(self, shape, dtype, prealloc):
        self.device = cuda.get_current_device()
        self.freelist = deque()
        self.events = {}
        for i in range(prealloc):
            gpumem = cuda.device_array(shape=shape, dtype=dtype)
            self.freelist.append(gpumem)
            self.events[gpumem] = cuda.event(timing=False)

    def get(self, stream=0):
        assert self.freelist
        gpumem = self.freelist.popleft()
        evnt = self.events[gpumem]
        if not evnt.query(): # not ready?
            # querying is faster then waiting
            evnt.wait(stream=stream) # future works must wait
        return gpumem

    def free(self, gpumem, stream=0):
        evnt = self.events[gpumem]
        evnt.record(stream=stream)
        self.freelist.append(gpumem)

