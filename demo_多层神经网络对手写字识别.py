# 1.导包
import paddle.fluid as fluid
import paddle


# 定义测试逻辑函数
def train_test(exe, feeder, reader, fetch_list, program):
    """

    :param exe:执行器
    :param feeder:数据与网络的关系
    :param reader:测试reader（数据读取器）
    :param fetch_list:需要运行的张量
    :param program:测试主进程
    :return:test_avg_loss,test_avg_acc
    """
    # 准确率高，损失小代表模型好
    # 如何将一件件损失批次----》整个测试集损失？
    # 将每个损失批次相加除以批次数量得到平均损失
    # 将每个准确率批次相加除以批次数量得到平均准确率
    all_avg_loss = 0
    all_acc = 0
    betch_num = 0
    for data in reader():
        # 返回每个批次的损失和准确率
        avg_loss_val, acc_val = exe.run(
            program=program,
            feed=feeder.feed(data),
            fetch_list=fetch_list,
        )
        all_avg_loss += avg_loss_val[0]
        all_acc += acc_val
        betch_num += 1
        # print(avg_loss_val)
        # print(acc_val)
    # 计算每个测试集的平均损失
    test_avg_loss = all_avg_loss / betch_num
    test_avg_acc = all_acc / betch_num

    return test_avg_loss, test_avg_acc


# 2.数据处理 ----- MNIST手写字已经经过了数据处理

# 3.构建reder ----paddlepaddle里面自动写好了reader

# 4.构建训练场所
place = fluid.CPUPlace()

# 5.配置网络结构
# 两个数据层
# 特征值数据层
# shape图像的三阶张量形式
img = fluid.layers.data(name="img", shape=[1, 28, 28], dtype="float32")
# 目标值数据层
# shape目标值的张量形式
label = fluid.layers.data(name="label", shape=[1], dtype="int64")

# 多层感知机模型---两个隐层--一个输出层 ---fc网络（全连接网络）
# 第一个隐层
# size神经元个数
h1 = fluid.layers.fc(input=img, size=128, act="relu", name="h1")

# 第二个隐层
h2 = fluid.layers.fc(input=h1, size=100, act="relu", name="h2")

# 输出层
y_predict = fluid.layers.fc(
    input=h2,
    size=10,  # 神经元个数，必须与分类的类别一致
    act="softmax",  # 多分类激活层的激活函数必须是softmax
    name="y_predict"
)

# 6.计算交叉熵损失
loss = fluid.layers.cross_entropy(input=y_predict, label=label)

# 计算批次的平均 交叉熵损失
avg_loss = fluid.layers.mean(loss)

# 7.计算准确率
acc = fluid.layers.accuracy(input=y_predict, label=label)

# 8.定义优化器
sgd_optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.1)
# 来指定优化交叉熵损失
sgd_optimizer.minimize(avg_loss)

# 9.定义数据与网络的关系
feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

# 10.定义执行器
# 定义训练执行器
exe_train = fluid.Executor(place=place)
# 定义测试执行器
exe_test = fluid.Executor(place=place)

# 11.初始化网络参数
# 生成参数初始化进程
startup_program = fluid.default_startup_program()

# 执行参数初始化进程
exe_train.run(startup_program)

# 12.生成训练主进程，测试主进程
train_program = fluid.default_main_program()
# 克隆训练主进程并告诉计算机克隆出来的进程是用来测试主进程的
test_program = train_program.clone(for_test=True)

# 13.生成reader，测试reader
# 先构建缓冲区，在缓冲区内部打乱样本顺序，拿出一个批次
train_reder = paddle.batch(
    reader=paddle.reader.shuffle(
        reader=paddle.dataset.mnist.train(),
        buf_size=50,  # 缓冲区大小
    ),
    batch_size=10,  # 每批次训练数据为10个样本
)
test_reader = paddle.batch(
    reader=paddle.reader.shuffle(
        reader=paddle.dataset.mnist.test(),
        buf_size=50,

    ),
    batch_size=10,  # 与训练reader批次数量相同
)

# 14.训练--双层循环

loop_num = 5
# 定义计数器--用于记录训练的次数
stop = 0
# 定义一个开关，来控制循环
flag = False
for loop in range(loop_num):
    for data in train_reder():
        avg_loss_val, acc_val = exe_train.run(
            program=train_program,  # 训练主进程
            feed=feeder.feed(data),  # 给程序传递真正的数据
            fetch_list=[avg_loss, acc],  # 传递需要运行的张量
        )
        # 每训练10次打印一下结果
        if stop % 10 == 0 and stop != 0:
            print("第%d次，损失为：%f，准确率为：%f" % (
                stop,
                avg_loss_val[0],
                acc_val[0]
            ))
        # 在测试集中集中测试
        # 没隔100步在测试集中进行测试
        if stop % 100 == 0 and stop != 0:
            test_avg_loss, test_avg_acc = train_test(
                exe=exe_test,
                reader=test_reader,
                program=test_program,
                fetch_list=[avg_loss, acc],
                feeder=feeder
            )
            print("***第 %d 次 测试集损失为：%f，测试集准确率为：%f***" % (
                stop,
                test_avg_loss,
                test_avg_acc,
            ))
            # 如果整个测试集损失<0.05 and 准确率>0.95此时认为训练结束
            if test_avg_loss < 0.05 and test_avg_acc > 0.95:
                print("最终测试结果损失为：%f，最终测试结果准确率为：%f" % (
                    test_avg_loss,
                    test_avg_acc,
                ))
                # 如果需要退出
                flag = True
            # 得到结果退出循环
                break
        # 每训练一次计数器+1
        stop += 1

    if flag:
        break
