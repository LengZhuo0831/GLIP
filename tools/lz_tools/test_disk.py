
import time
import os

import numpy

def write_a_log_file(lines=10,sleep_val=10):
    path = '/root/share/lengz/test_disk/'
    t = time.localtime()
    name = f'{t.tm_year}-{t.tm_mon}-{t.tm_mday}--{t.tm_hour}:{t.tm_min}:{t.tm_sec}.log'
    print('writing',name)
    with open(os.path.join(path,name),'w') as f:
        for i in range(lines):
            t = time.localtime()
            content=f'date: {t.tm_year}-{t.tm_mon}-{t.tm_mday}\ntime:{t.tm_hour}:{t.tm_min}:{t.tm_sec}\n'
            f.write(content)
            # print(content)
            time.sleep(sleep_val)
        f.close()
    # print(str(t))
    
def write_numpy_data(size=(100,100),sleep_val=10):
    path = '/root/share/lengz/test_disk/'
    t = time.localtime()
    name = f'{t.tm_year}-{t.tm_mon}-{t.tm_mday}--{t.tm_hour}:{t.tm_min}:{t.tm_sec}.npy'
    print('saving', name)
    data = numpy.random.randn(*size)
    numpy.save(os.path.join(path,name), data)
    time.sleep(sleep_val)

# write_a_log_file(15,1)
# write_numpy_data(size=(100,100,20),sleep_val=10)


def main():
    print('testing writing on disk')
    # 这个程序多跑一下吧，大概，跑一天；86400秒
    # 每100秒生成一个log和一个numpy(100,100,20),1.5M
    for i in range(864):
        print(i,'/','864')
        write_a_log_file(lines=50,sleep_val=50)
        write_numpy_data(size=(100,100,20),sleep_val=50)
    
    
if __name__=='__main__':
    main()