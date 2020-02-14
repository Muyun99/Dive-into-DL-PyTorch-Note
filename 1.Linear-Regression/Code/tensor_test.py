import torch
import time 

class Timer():
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.times.append(time.time() - self.start_time)
        return self.times[-1]

    def time_sum(self):
        return sum(self.times)

    def time_avg(self):
        return sum(self.times) / len(self.times)

if __name__ == "__main__":
    n = 1000
    a = torch.ones(n)
    b = torch.ones(n)
    c = torch.zeros(n)
    
    timer = Timer()
    timer.start()
    for i in range(n):
        c[i] = a[i] + b[i]
    print("%.5f sec" %timer.stop())

    timer.start()
    c = a + b
    print("%.5f sec" %timer.stop())
