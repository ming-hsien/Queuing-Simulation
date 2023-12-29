from sympy import *
import random
import numpy as np
import matplotlib.pyplot as plt

# Interarrival rate
lamda = 10
# Service rate
Mu = [10, 20]
Prob = [0.25,0.75]
# Packet num
packetNums = 10000

class QueueingSystem():
    def __init__(self, arrival_rate, service_rate):
        self.arrival_rate = arrival_rate
        self.service_rate1 = service_rate[0]
        self.service_rate2 = service_rate[1]
        self.s1_probability = Prob[0]
        self.s2_probability = Prob[1]
        self.WaitingTime_mse = 0
        self.SystemTime_mse = 0
        self.queue = []
        self.WaitingTime  = []
        self.ServiceTime = []
        self.SystemTime = []
        self.SystemTime_theo = []
        self.WaitingTime_theo = []

    def packet_arrival(self):
        execution_time = 0
        for i in range(packetNums):
            execution_time = np.random.exponential(1 / self.arrival_rate)
            execution_time = execution_time + self.queue[i - 1] if i >= 1 else execution_time
            self.queue.append(execution_time)

    def packet_service(self):
        execution_time = 0
        service_time = None
        for pkid in range(packetNums):
            if execution_time < self.queue[pkid]:
                execution_time = self.queue[pkid]
            a = random.randrange(100)
            service_time = np.random.exponential(1 / self.service_rate1) if a < (100 * self.s1_probability) else np.random.exponential(1 / self.service_rate2)
            self.WaitingTime.append(execution_time - self.queue[pkid])
            self.ServiceTime.append(service_time)
            execution_time += service_time

    def get_WaitingTime_theo(self):
        lo = Prob[0]*(lamda / Mu[0]) + Prob[1]*(lamda / Mu[1])
        s = Symbol('s')
        t = Symbol('t')
        B_Laplace_s = Prob[0] * (Mu[0] / (s + Mu[0])) + Prob[1] * (Mu[1] / (s + Mu[1]))
        # W_Laplace_s / s 相當於積分,因此後面不需要積分 (By Dr. Stone)
        W_Laplace_s = (1 - lo) / (s - lamda + lamda * B_Laplace_s)
        w_y = inverse_laplace_transform(W_Laplace_s, s, t)
        self.WaitingTime_theo = np.unique(sorted(self.WaitingTime))
        for i, wt in enumerate(self.WaitingTime_theo):
            self.WaitingTime_theo[i]  = w_y.subs(t, wt) * 100
            # self.WaitingTime_theo[i] = integrate(w_y, t, 0, wt)
 
    def get_SystemTime_theo(self):
        lo = Prob[0]*(lamda / Mu[0]) + Prob[1]*(lamda / Mu[1])
        s = Symbol('s')
        t = Symbol('t')
        B_Laplace_s = Prob[0] * (Mu[0] / (s + Mu[0])) + Prob[1] * (Mu[1] / (s + Mu[1]))
        W_Laplace_s = s * (1 - lo) / (s - lamda + lamda * B_Laplace_s)
        S_Laplace_s = B_Laplace_s * W_Laplace_s
        S_y = inverse_laplace_transform(S_Laplace_s, s, t)
        S_integrate = integrate(S_y)
        self.SystemTime_theo = np.unique(sorted(self.SystemTime))
        for i, st in enumerate(self.SystemTime_theo):
            self.SystemTime_theo[i] = S_integrate.subs(t, st) * 100
            # self.SystemTime_theo[i] = Integral(S_y,(t,0,st))

    def plot_Theoretical_WaitingTime_CDF(self):
        plot_x = np.unique(sorted(self.WaitingTime))
        plot_y = self.WaitingTime_theo
        plt.plot(plot_x, plot_y, 'r', label = "Theoreotical")
    
    def plot_Theoretical_SystemTime_CDF(self):
        plot_x = np.unique(sorted(self.SystemTime))
        plot_y = self.SystemTime_theo
        plt.plot(plot_x, plot_y, 'r', label = "Theoreotical")

    def plot_WaitingTime_CDF(self):
        self.WaitingTime.sort()
        plot_x = np.unique(self.WaitingTime)
        plot_y = []
        for i,unique_x in enumerate(plot_x):
            nums_u = self.WaitingTime.count(unique_x)
            if i == 0:
                plot_y.append(nums_u / packetNums * 100)
            else:
                plot_y.append(plot_y[i - 1] + (nums_u / packetNums) * 100)
        for i, w in enumerate(plot_y):
            self.WaitingTime_mse += (w - self.WaitingTime_theo[i])*(w - self.WaitingTime_theo[i])
        print("WaitingTime MSE:",self.WaitingTime_mse / packetNums)
        plt.plot(plot_x, plot_y, 'b', label = "Simulation")
        self.plot_Theoretical_WaitingTime_CDF()
        plt.title('Scenario 2 - Waiting Time Distribution (λ={})'.format(lamda))
        plt.xlabel('y (Unit Time)')
        plt.ylabel('W(y) CDF (%)')
        plt.grid(True)
        plt.xlim(xmin = 0)
        plt.ylim(0,100)
        plt.legend(loc = 'upper right', fontsize = '9')
        plt.savefig('scenario2_w.png')

    def plot_SystemTime_CDF(self):
        plt.cla()
        self.SystemTime.sort()
        plot_x = np.unique(self.SystemTime)
        plot_y = []
        for i, unique_x in enumerate(plot_x):
            nums_u = self.SystemTime.count(unique_x)
            if i == 0:
                plot_y.append(nums_u / packetNums * 100)
            else:
                plot_y.append(plot_y[i - 1] + (nums_u / packetNums) * 100)
        for i, w in enumerate(plot_y):
            self.SystemTime_mse += (w - self.SystemTime_theo[i])*(w - self.SystemTime_theo[i])
        print("SystemTime MSE:",self.SystemTime_mse / packetNums)
        plt.plot(plot_x, plot_y, 'b', label = "Simulation")
        self.plot_Theoretical_SystemTime_CDF()
        plt.title('Scenario 2 - System Time Distribution (λ={})'.format(lamda))
        plt.xlabel('y (Unit Time)')
        plt.ylabel('S(y) CDF (%)')
        plt.grid(True)
        plt.xlim(xmin = 0)
        plt.ylim(0, 100)
        plt.legend(loc = 'upper right', fontsize = '9')
        plt.savefig('scenario2_s.png')


    def run(self):
        self.packet_arrival()
        self.packet_service()
        self.get_WaitingTime_theo()
        self.SystemTime = list(np.add(self.WaitingTime, self.ServiceTime))
        self.get_SystemTime_theo()
        

if __name__ == '__main__':
    QS = QueueingSystem(lamda, Mu)
    QS.run()
    QS.plot_WaitingTime_CDF()
    QS.plot_SystemTime_CDF()
