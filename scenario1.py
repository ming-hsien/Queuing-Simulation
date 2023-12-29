import numpy as np
import matplotlib.pyplot as plt

# Interarrival rate
lamda = 8
# Service rate
Mu = 10
# Packet num
packetNums = 10000

class QueueingSystem():
    def __init__(self, arrival_rate, service_rate):
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
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
            service_time = np.random.exponential(1 / self.service_rate)
            self.WaitingTime.append(execution_time - self.queue[pkid])
            self.ServiceTime.append(service_time)
            execution_time += service_time

    def get_WaitingTime_theo(self):
        lo = lamda / Mu
        self.WaitingTime_theo = np.unique(sorted(self.WaitingTime))
        for i, wt in enumerate(self.WaitingTime_theo):
            self.WaitingTime_theo[i]  = (1 - lo * np.exp(-Mu * (1-lo) * wt)) * 100

    def get_SystemTime_theo(self):
        lo = lamda / Mu
        self.SystemTime_theo = np.unique(sorted(self.SystemTime))
        for i, st in enumerate(self.SystemTime_theo):
            self.SystemTime_theo[i] = (1 - np.exp(-Mu * (1 - lo) * st)) * 100


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
        plt.title('Scenario 1 - Waiting Time Distribution (λ={}, μ={})'.format(lamda, Mu))
        plt.xlabel('y (Unit Time)')
        plt.ylabel('W(y) CDF (%)')
        plt.grid(True)
        plt.xlim(xmin = 0)
        plt.ylim(0,100)
        plt.legend(loc = 'upper right', fontsize = '9')
        plt.savefig('scenario1_w.png')

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
        plt.title('Scenario 1 - System Time Distribution (λ={}, μ={})'.format(lamda, Mu))
        plt.xlabel('y (Unit Time)')
        plt.ylabel('S(y) CDF (%)')
        plt.grid(True)
        plt.xlim(xmin = 0)
        plt.ylim(0, 100)
        plt.legend(loc = 'upper right', fontsize = '9')
        plt.savefig('scenario1_s.png')


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
