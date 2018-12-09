class add_node():
    '''加法操作'''
    def __init__(self,x,y):
        self.res=x+y
        self.grad_x=1
        self.grad_y=1

class cheng_node():
    '''乘法操作'''
    def __init__(self,x,y):
        self.res=x*y
        self.grad_x=y
        self.grad_y=x

class pingfang_node():
    '''平方操作'''
    def __init__(self,x):
        self.res=x**2
        self.grad_x=2*x

class cell():
    def __init__(self,x,w):
        self.grad=[]
        self.res=0     # 输出默认0
        for i in range(len(x)):
            self.res += x[i]*w[i]
            self.grad.append(x[i])    # x[i]为导数 添加到列表里
        # 加上最后一个w[]   w比x多一个 偏置
        self.res += w[len(x)]
        self.grad.append(1)


x=[[0.1,0.8],[0.8,0.2]]
y=[3,2]
w=[0.1,0.2,0.3]
lr=0.01

for i in range(1000):
    # 得到两个输出
    out_0=cell(x[0],w)
    out_1=cell(x[1],w)
    print('out',[out_0.res,out_1.res])

    # loss=(out-y)^2=(out0-y0)^2+(out1-y1)^2
    an_0=add_node(out_0.res,-y[0])
    pfn_0=pingfang_node(an_0.res)
    an_1 = add_node(out_1.res, -y[1])
    pfn_1 = pingfang_node(an_1.res)

    an_all=add_node(pfn_0.res,pfn_1.res)
    print('loss',an_all.res)

    # 反向传播 找关于w的
    td0=an_all.grad_x * pfn_0.grad_x * an_0.grad_x * out_0.grad[0] + an_all.grad_y * pfn_1.grad_x * an_1.grad_x * out_1.grad[0]
    td1=an_all.grad_x * pfn_0.grad_x * an_0.grad_x * out_0.grad[1] + an_all.grad_y * pfn_1.grad_x * an_1.grad_x * out_1.grad[1]
    td2=an_all.grad_x * pfn_0.grad_x * an_0.grad_x * out_0.grad[2] + an_all.grad_y * pfn_1.grad_x * an_1.grad_x * out_1.grad[2]

    w[0] -= td0 * lr
    w[1] -= td1 * lr
    w[2] -= td2 * lr


