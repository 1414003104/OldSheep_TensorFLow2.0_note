import tensorflow as tf
import timeit
#测试是否使用了GPU
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

with tf.device('/cpu:0'):
    cpu_a=tf.random.normal([10000,1000])#矩阵新建
    cpu_b=tf.random.normal([1000,2000])
    print(cpu_a.device,cpu_b.device)

with tf.device('/gpu:0'):
    gpu_a = tf.random.normal([10000, 1000])
    gpu_b = tf.random.normal([1000, 2000])
    print(gpu_a.device, gpu_b.device)

def cpu_run():
    with tf.device('/cpu:0'):
        c=tf.matmul(cpu_a,cpu_b)#矩阵乘法
    return c

def gpu_run():
    with tf.device('/gpu:0'):
        c=tf.matmul(gpu_a,gpu_b)
    return c

#warm up 热身时间
cpu_time=timeit.timeit(cpu_run,number=10)
gpu_time=timeit.timeit(gpu_run,number=10)
print('warmup:',cpu_time,gpu_time)

#正式开始测试
cpu_time=timeit.timeit(cpu_run,number=10)
gpu_time=timeit.timeit(gpu_run,number=10)
print('run time',cpu_time,gpu_time)