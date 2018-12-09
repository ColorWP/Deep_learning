'''
以简单小例子   最大化的了解   反向传播
反向传播是什么？
说明白点 就是数学上的求导
高深一点 就是链式求导法则

'''

x=[1,2]
y=0.5

w=[0.1,0.2,0.3]

def foward(x,w):
    out=x[0]*w[0]+x[1]*w[1]+w[2]
    return out

lr=0.01

for i in range(100):
    out =foward(x,w)

    loss=(out-y)**2
    print(loss)

    td=[x[0]*2*(out-y),x[1]*2*(out-y),2*(out-y)]

    w[0] += -td[0] * lr
    w[1] += -td[1] * lr
    w[2] += -td[2] * lr

print(foward(x,w))


