import functools
import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # we can use a decorator function to convert operation inputs to Value objects
    # which makes the code a bit cleaner
    def input_to_value():
        def decorator(f):
            @functools.wraps(f)
            def wrapped_input_to_value(s, o, *args):
                o = o if isinstance(o, Value) else Value(o)
                return f(s, o, *args)
            return wrapped_input_to_value
        return decorator
    
    # we can also use a decorator to set out._backward
    def set_out_backward():
        def decorator(f):
            @functools.wraps(f)
            def wrapped_set_out_backward(*args):
                out, back = f(*args)
                out._backward = back
                return out
            return wrapped_set_out_backward
        return decorator


    def __repr__(self):
        return f"Value(data={self.data})"

    @input_to_value()
    @set_out_backward()
    def __add__(self, other):  # exactly as in the video
        out = Value(self.data + other.data, (self, other), _op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        return out, _backward
    
    # ------
    # re-implement all the other functions needed for the exercises below
    # your code here

    @input_to_value()
    @set_out_backward()
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        return out, _backward

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    

    @set_out_backward()
    def __pow__(self, other):
        out = Value(self.data ** other, (self,), _op='**')
        
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        return out, _backward
    
    @set_out_backward()
    def exp(self):
        out = Value(math.exp(self.data), (self,), _op='exp')

        def _backward():
            self.grad += (out.data) * out.grad

        return out, _backward
    
    def __truediv__(self, other):
        return self * (other ** -1)

    @set_out_backward()
    def log(self):
        out = Value(math.log(self.data), (self,), _op='log')

        def _backward():
            self.grad += (1 / self.data) * out.grad

        return out, _backward
    
    @set_out_backward()
    def sin(self):
        out = Value(math.sin(self.data), (self,), _op='sin')
        
        def _backward():
            self.grad += (math.cos(self.data)) * out.grad
        
        return out, _backward

    @set_out_backward()
    def cos(self):
        out = Value(math.cos(self.data), (self,), _op='cos')
        
        def _backward():
            self.grad += (-math.sin(self.data)) * out.grad
        
        return out, _backward
    
    @set_out_backward()
    def tan(self):
        out = Value(math.tan(self.data), (self,), _op='tan')

        def _backward():
            self.grad += (1 / (math.cos(self.data) ** 2)) * out.grad
        
        return out, _backward
    
    @set_out_backward()
    def sinh(self):
        out = Value(math.sinh(self.data), (self,), _op='sinh')

        def _backward():
            self.grad += (math.cosh(self.data)) * out.grad
        
        return out, _backward
    

    @set_out_backward()
    def cosh(self):
        out = Value(math.cosh(self.data), (self,), _op='cosh')

        def _backward():
            self.grad += (math.sinh(self.data)) * out.grad
        
        return out, _backward

    @set_out_backward()
    def tanh(self):
        out = Value(math.tanh(self.data), (self,), _op='tanh')

        def _backward():
            self.grad += (1 / (math.cosh(self.data) ** 2)) * out.grad
        
        return out, _backward

    @set_out_backward()
    def relu(self):
        out = Value(max(0, self.data), (self,), _op='relu')
        
        def _backward():
            self.grad += out.grad if out.data > 0 else 0

        return out, _backward

    def sigmoid(self):
        return 1 / (1 + (-self).exp())

    # -----
    # Reverse Methods:
    # (need to convert other here to prevent infinite recursion)
    @input_to_value()
    def __radd__(self, other):
        return other + self
    @input_to_value()
    def __rsub__(self, other):
        return other - self
    @input_to_value()
    def __rmul__(self, other):
        return other * self
    @input_to_value()
    def __rtruediv__(self, other):
        return other / self
    # -----

    # -----
    # Named Methods:
    def add(self, other):
        return self + other
    def sub(self, other):
        return self - other
    def mul(self, other):
        return self * other
    def div(self, other):
        return self / other
    def pow(self, other):
        return self ** other
    # -----
    

    def backward(self):  # exactly as in video
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
