# %% 最小二乘法非线性函数拟合返回参数和拟合优度
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import pandas as pd

# 定义拟合函数
def func(p, x):
    a, b, c = p
    return a + b * x ** c #power 

# 定义误差函数
def error(p, x, y):
    return func(p, x) - y

# 生成数据
df = pd.read_excel('Interval and Z2.xlsx', sheet_name='6-2-yz-1', header=0, usecols='BL:BO', nrows=1001)
Iv = df.values[0:16,0]
z2 = df.values[0:16,2]
Iv_fit = np.linspace(0.05,10,200) 

# 初始参数
p0 = [0, 0, 0]

# 最小二乘法拟合
para, cov = leastsq(error, p0, args=(Iv, z2))

# 计算拟合优度值
ss_err = ((func(para,Iv)-z2.mean() )** 2).sum()
ss_tot = ((z2 - z2.mean()) ** 2).sum()
r_squared = ss_err / ss_tot

# 绘制图像
plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '24'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig2 = plt.figure(figsize=(12,8), edgecolor='white', dpi=200)
ax = plt.subplot(111)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_xlim(-1,11)
ax.set_ylim(0,0.8)
ax.set_xticks(np.arange(0,11,1))
ax.set_yticks(np.arange(0,0.8,0.1))
ax.set_ylabel('Z2/mm')
ax.set_xlabel('Interval/mm')
# ax.set_title('Fruit supply by kind and color')
ax.grid(True,ls='--')
ax.scatter(Iv,z2,marker='o',color='red',s=50,label='test')
ax.plot(Iv_fit, para[0]+para[1]*Iv_fit**para[2], color='red', linestyle='solid', lw=2,label='fit')
ax.legend()

plt.show()
print(para)
print(f"拟合优度值: {r_squared:.6f}")
# %% 绘制轮廓线 **************************************************************
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_excel('profile_data.xlsx', sheet_name='6', header=0, usecols='A:J', nrows=1002)
x = df.values[0:1001,0]
y = df.values[0:1001,1]

sampling_interval = 1
interval_space = sampling_interval/0.05

x1 = x[0::int(interval_space)]
y1 = y[0::int(interval_space)]

plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '24'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig1 = plt.figure(figsize=(24,2), edgecolor='white', dpi=200)
ax = plt.subplot(111)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_xlim(-5,55)
ax.set_ylim(-3,3)
ax.set_xticks(np.arange(0,55,5))
ax.set_yticks(np.arange(-3,3,1))
ax.set_ylabel('y/mm')
ax.set_xlabel('x/mm')
ax.grid(True,ls='--')
ax.scatter(x1,y1,marker='^',color='blue',s=10,label='profile')
ax.plot(x1,y1,color='blue',lw=2,label='profile')

plt.savefig('pf_6-2-yz.svg')
plt.show()
# %%  计算Grasselli值
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import leastsq

def ag_cal(x_G, y_G, ag_space_G):

    ag_list = []

    for i in range(len(x_G)-1):

        ag = np.rad2deg(np.arctan((y_G[i+1]-y_G[i])/(x_G[i+1]-x_G[i])))
        ag_list.append(ag)
        i = i+1

    ag_list = np.array(ag_list)

    count_po_list = []

    for i in range(int(90/ag_space_G)):

        count = np.sum(ag_list >= ag_space_G*i)
        count_po_list.append(count)
        i = i+1

    count_po_list = np.array(count_po_list)

    count_ne_list = []

    for i in range(int(90/ag_space_G)):

        count = np.sum(ag_list <= ag_space_G*(-i))
        count_ne_list.append(count)
        i = i+1

    count_ne_list = np.array(count_ne_list)

    return ag_list, count_po_list, count_ne_list

def fit_Projection_vs_ag(para_1, para_2, x, y):

    def func(para_1, para_2, p, x):
        c = p
        return para_1 * ((para_2-x)/para_2)**c  #Grasselli

    # 定义误差函数
    def error(p, x, y):
        return func(para_1, para_2, p, x) - y

    # 初始参数
    p0 = [1]

    # 最小二乘法拟合
    para_fit, cov = leastsq(error, p0, args=(x, y))

    # 计算拟合优度值
    ss_err = ((func(para_1, para_2, para_fit, x)-y)** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared =1 - ss_err / ss_tot

    return para_fit, r_squared

def sampling_interval_vs_Grasselli(sampling_interval,ag_space,x,y):

    interval_space = sampling_interval/0.05 

    x1 = x[0::int(interval_space)]
    y1 = y[0::int(interval_space)]

    ag_list, count_po_list, count_ne_list= ag_cal(x1,y1,ag_space)

    ag_list_po = ag_list[ag_list > 0]
    ag_list_ne = np.absolute(ag_list[ag_list < 0])

    po_lg = count_po_list / (len(x1)-1)
    ne_lg = count_ne_list / (len(x1)-1)

    po_lg = po_lg[po_lg != 0] 
    ne_lg = ne_lg[ne_lg != 0] 

    ag_x_po = np.linspace(0,int(len(po_lg))*ag_space-1,int(len(po_lg)))
    ag_x_ne = np.linspace(0,int(len(ne_lg))*ag_space-1,int(len(ne_lg)))

    G_c_po, r2_po = fit_Projection_vs_ag(np.max(po_lg), np.max(ag_list_po), ag_x_po, po_lg)
    G_c_ne, r2_ne = fit_Projection_vs_ag(np.max(ne_lg), np.max(ag_list_ne), ag_x_ne, ne_lg)

    G_value_po = np.max(ag_list_po)/(G_c_po + 1)
    G_value_ne = np.max(ag_list_ne)/(G_c_ne + 1)

    return po_lg, ag_x_po, ag_list_po, G_c_po, r2_po, G_value_po, ne_lg, ag_x_ne, ag_list_ne, G_c_ne, r2_ne, G_value_ne

input_data = ['15', 3, 0.05, int(1), 1000]

df = pd.read_excel('profile_data.xlsx', sheet_name=input_data[0], header=0, usecols='A:F', nrows=1001)
x = np.array(df.values[0:input_data[4],0])
y = np.array(df.values[0:input_data[4],input_data[1]])

sampling_interval = input_data[2]
ag_space = input_data[3]

po_lg, ag_x_po, ag_list_po, G_c_po, r2_po, G_value_po, ne_lg, ag_x_ne, ag_list_ne, G_c_ne, r2_ne, G_value_ne = sampling_interval_vs_Grasselli(sampling_interval,ag_space,x,y)

# print(f"forward  lg: {np.max(po_lg):.3f},", f"max ag:{np.max(ag_list_po):.3f},", f"c:{G_c_po[0]:.3f},", f"fitting: {r2_po:.3f},",f"G_value:{G_value_po[0]:.3f}")
# print(f"backward lg: {np.max(ne_lg):.3f},", f"max ag:{np.max(ag_list_ne):.3f},", f"c:{G_c_ne[0]:.3f},", f"fitting: {r2_ne:.3f},",f"G_value:{G_value_ne[0]:.3f}")

plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '20'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig = plt.figure(figsize=(12,8), edgecolor='white', dpi=200)
ax = plt.subplot(111)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_xlim(-1,90)
ax.set_ylim(0,0.7)
ax.set_xticks(np.arange(0,91,5))
ax.set_yticks(np.arange(0,0.8,0.1))
ax.set_ylabel('$A_0$')
ax.set_xlabel('Angle of inclination/°')
ax.grid(True,ls='--')

ax.bar(ag_x_po, po_lg, color='red', alpha = 0.5)
ax.bar(ag_x_ne, ne_lg, color='green', alpha = 0.5)
ax.scatter(ag_x_po, po_lg,marker='o',color='red',s=50,label='forward test')
ax.scatter(ag_x_ne, ne_lg,marker='o',color='green',s=50,label='backward test')
ax.plot(ag_x_po, np.max(po_lg) * ((np.max(ag_list_po)-ag_x_po)/np.max(ag_list_po))**G_c_po, color='red', linestyle='solid', lw=2,label='forward fit')
ax.plot(ag_x_ne, np.max(ne_lg) * ((np.max(ag_list_ne)-ag_x_ne)/np.max(ag_list_ne))**G_c_ne, color='green', linestyle='solid', lw=2,label='backward fit')
ax.legend(loc=(65/90,0.3/0.7))

ax.text(25,0.53,f'Specimen No.{input_data[0]} Profile No.{input_data[1]} Interval:{input_data[2]} Ag_spcae:{input_data[3]}')
ax.text(50,0.01,f'Forward \nlg:{np.max(po_lg):.3f} \nAg_max:{np.max(ag_list_po):.3f} \nc_value:{G_c_po[0]:.3f} \nG_value:{G_value_po[0]:.3f} \n$R^2$:{r2_po:.3f}')
ax.text(70,0.01,f'Backward \nlg:{np.max(ne_lg):.3f} \nAg_max:{np.max(ag_list_ne):.3f} \nc_value:{G_c_ne[0]:.3f} \nG_value:{G_value_ne[0]:.3f} \n$R^2$:{r2_ne:.3f}')

pfplot = fig.add_axes([.2, .78, .65, .08],facecolor='lightyellow')
pfplot.tick_params(labelsize=12)
pfplot.set_xlim(-1,51)
pfplot.set_ylim(-3,3)
pfplot.set_xlabel('x/mm', fontsize=12)
pfplot.set_ylabel('y/mm', fontsize=12)
pfplot.scatter(x,y,marker='^',color='blue',s=2,label='profile')
pfplot.plot(x,y,color='blue',lw=1,label='profile')

plt.savefig('./fig/G_value/A0_'+str(input_data[0])+'_'+str(input_data[1])+'_'+str(input_data[2])+'.pdf')
plt.show()

# %%  计算不同sampling interval的Grasselli值
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import leastsq

def ag_cal(x_G, y_G, ag_space_G):

    ag_list = []

    for i in range(len(x_G)-1):

        ag = np.rad2deg(np.arctan((y_G[i+1]-y_G[i])/(x_G[i+1]-x_G[i])))
        ag_list.append(ag)
        i = i+1

    ag_list = np.array(ag_list)

    count_po_list = []

    for i in range(int(90/ag_space_G)):

        count = np.sum(ag_list >= ag_space_G*i)
        count_po_list.append(count)
        i = i+1

    count_po_list = np.array(count_po_list)

    count_ne_list = []

    for i in range(int(90/ag_space_G)):

        count = np.sum(ag_list <= ag_space_G*(-i))
        count_ne_list.append(count)
        i = i+1

    count_ne_list = np.array(count_ne_list)

    return ag_list, count_po_list, count_ne_list

def fit_Projection_vs_ag(para_1, para_2, x, y):

    def func(para_1, para_2, p, x):
        c = p
        return para_1 * ((para_2-x)/para_2)**c  #Grasselli

    # 定义误差函数
    def error(p, x, y):
        return func(para_1, para_2, p, x) - y

    # 初始参数
    p0 = [1]

    # 最小二乘法拟合
    para_fit, cov = leastsq(error, p0, args=(x, y))

    # 计算拟合优度值
    ss_err = ((func(para_1, para_2, para_fit, x)-y)** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared =1 - ss_err / ss_tot

    return para_fit, r_squared

def sampling_interval_vs_Grasselli(sampling_interval,ag_space,x,y):

    interval_space = sampling_interval/0.05 

    x1 = x[0::int(interval_space)]
    y1 = y[0::int(interval_space)]

    ag_list, count_po_list, count_ne_list= ag_cal(x1,y1,ag_space)

    ag_list_po = ag_list[ag_list > 0]
    ag_list_ne = np.absolute(ag_list[ag_list < 0])

    po_lg = count_po_list / (len(x1)-1)
    ne_lg = count_ne_list / (len(x1)-1)

    po_lg = po_lg[po_lg != 0] 
    ne_lg = ne_lg[ne_lg != 0] 

    ag_x_po = np.linspace(0,int(len(po_lg))*ag_space-1,int(len(po_lg)))
    ag_x_ne = np.linspace(0,int(len(ne_lg))*ag_space-1,int(len(ne_lg)))

    G_c_po, r2_po = fit_Projection_vs_ag(np.max(po_lg), np.max(ag_list_po), ag_x_po, po_lg)
    G_c_ne, r2_ne = fit_Projection_vs_ag(np.max(ne_lg), np.max(ag_list_ne), ag_x_ne, ne_lg)

    G_value_po = np.max(ag_list_po)/(G_c_po + 1)
    G_value_ne = np.max(ag_list_ne)/(G_c_ne + 1)

    return po_lg, ag_x_po, ag_list_po, G_c_po, r2_po, G_value_po, ne_lg, ag_x_ne, ag_list_ne, G_c_ne, r2_ne, G_value_ne

input_data = ['15', 3, int(1), 1000]

df = pd.read_excel('profile_data.xlsx', sheet_name=input_data[0], header=0, usecols='A:F', nrows=1001)
x = np.array(df.values[0:input_data[3],0])
y = np.array(df.values[0:input_data[3],input_data[1]])

ag_space = input_data[2]

Inter = np.linspace(0.05,10,200)
G_value_po_list = []
G_value_ne_list = []

for i in range(len(Inter)):

    G_value_po = sampling_interval_vs_Grasselli(Inter[i], ag_space, x, y)[5]
    G_value_po_list.append(G_value_po[0])
    i = i + 1

G_value_po_list = np.array(G_value_po_list)

for i in range(len(Inter)):

    G_value_ne = sampling_interval_vs_Grasselli(Inter[i], ag_space, x, y)[11]
    G_value_ne_list.append(G_value_ne[0])
    i = i + 1

G_value_ne_list = np.array(G_value_ne_list)

# print(G_value_ne_list)

plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '20'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig = plt.figure(figsize=(12,8), edgecolor='white', dpi=200)
ax = plt.subplot(111)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_xlim(-1,11)
ax.set_ylim(0,12)
ax.set_xticks(np.arange(0,11,1))
ax.set_yticks(np.arange(0,13,1))
ax.set_ylabel('G_value')
ax.set_xlabel('Interval/mm')
ax.grid(True,ls='--')

ax.scatter(Inter, G_value_po_list, marker='o',color='red',s=50,label='po')
ax.scatter(Inter, G_value_ne_list, marker='s',color='green',s=50,label='ne')
# ax.plot(ag_x_po, np.max(po_lg) * ((np.max(ag_list_po)-ag_x_po)/np.max(ag_list_po))**G_c_po, color='red', linestyle='solid', lw=2,label='forward fit')
ax.legend(loc=(9/12,7/12))
ax.text(0,1,f'Specimen No.{input_data[0]} \nProfile No.{input_data[1]} \nAg_space:{input_data[2]}')

pfplot = fig.add_axes([.2, .78, .65, .08],facecolor='lightyellow')
pfplot.tick_params(labelsize=12)
pfplot.set_xlim(-1,51)
pfplot.set_ylim(-3,3)
pfplot.set_xlabel('x/mm', fontsize=12)
pfplot.set_ylabel('y/mm', fontsize=12)
pfplot.scatter(x,y,marker='^',color='blue',s=2,label='profile')
pfplot.plot(x,y,color='blue',lw=1,label='profile')

plt.savefig('./fig/G_value/Gvalue_'+str(input_data[0])+'_'+str(input_data[1])+'_'+str(input_data[2])+'.pdf')
plt.show()
# %%  计算z2值与间隔的关系
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import leastsq

def z2_cal(interval, x, y):

    space = interval/0.05
    k_arr=[]
    x1 = x[0::int(space)]
    y1 = y[0::int(space)]

    for i in range(len(x1)-1):

        k = ((y1[i+1]-y1[i])/(x1[i+1]-x1[i]))
        k_arr.append(k)

        i = i+1

    k_arr = np.array([k_arr]) 

    k_arr_po = k_arr[k_arr > 0]
    k_arr_ne = k_arr[k_arr < 0]

    z_2 = np.average(np.square(k_arr))**0.5
    z_2_po = np.average(np.square(k_arr_po))**0.5
    z_2_ne = np.average(np.square(k_arr_ne))**0.5

    return z_2, z_2_po, z_2_ne

def fit_z2_vs_interval(Iv,z2_list):

    def func(p, x):
        a, b = p
        return a + b * x #linear

    # 定义误差函数
    def error(p, x, y):
        return func(p, x) - y

    # 初始参数
    p0 = [0, 0]

    # 最小二乘法拟合
    para, cov = leastsq(error, p0, args=(Iv, z2_list))

    # 计算拟合优度值
    ss_err = ((func(para,Iv)-z2_list.mean() )** 2).sum()
    ss_tot = ((z2_list - z2_list.mean()) ** 2).sum()
    r_squared = ss_err / ss_tot

    return para, r_squared

input_data = ['15', 3, int(200), 1000]

df = pd.read_excel('profile_data.xlsx', sheet_name=input_data[0], header=0, usecols='A:F', nrows=1001)
x = np.array(df.values[0:input_data[3],0])
y = np.array(df.values[0:input_data[3],input_data[1]])

Inter = np.linspace(0.05,10,input_data[2])
z2_list = []
z2_list_po = []
z2_list_ne = []

for i in range(len(Inter)):

    z2 = z2_cal(Inter[i],x,y)[0]
    z2_list.append(z2)

    z2_po = z2_cal(Inter[i],x,y)[1]
    z2_list_po.append(z2_po)

    z2_ne = z2_cal(Inter[i],x,y)[2]
    z2_list_ne.append(z2_ne)

    i = i+1

z2_list = np.array(z2_list)

para, r_squared = fit_z2_vs_interval(Inter,z2_list)

# print(para,f"拟合优度值: {r_squared:.6f}")

plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '20'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig = plt.figure(figsize=(12,8), edgecolor='white', dpi=200)
ax = plt.subplot(111)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_xlim(-1,11)
ax.set_ylim(0,0.3)
ax.set_xticks(np.arange(0,11,1))
ax.set_yticks(np.arange(0,0.31,0.02))
ax.set_ylabel('$Z_2$/mm')
ax.set_xlabel('Interval/mm')
ax.grid(True,ls='--')

# ax.scatter(Inter, z2_list, marker='o', color='red', s=50,label='z2 all')
# ax.scatter(Inter, z2_list_po, marker='s', color='green', s=50,label='z2 po')
ax.scatter(Inter, z2_list, marker='^', color='blue', s=50,label='$Z_2$')
ax.plot(Inter, para[0]+para[1]*Inter, color='red', linestyle='solid', lw=2,label='fit')
ax.legend(loc=(9/12,18/30))
ax.text(0,0.02,f'Specimen No.{input_data[0]} \nProfile No.{input_data[1]} \nSI:{input_data[2]} \nIntercept={para[0]:.3f} \nSlope={para[1]:.3f} \n$R^2$={r_squared:.3f}')

pfplot = fig.add_axes([.2, .78, .65, .08],facecolor='lightyellow')
pfplot.tick_params(labelsize=12)
pfplot.set_xlim(-1,51)
pfplot.set_ylim(-3,3)
pfplot.set_xlabel('x/mm', fontsize=12)
pfplot.set_ylabel('y/mm', fontsize=12)
pfplot.scatter(x,y,marker='^',color='blue',s=2,label='profile')
pfplot.plot(x,y,color='blue',lw=1,label='profile')

plt.savefig('./fig/z2_interval/Z2_'+str(input_data[0])+'_'+str(input_data[1])+'_'+str(input_data[2])+'.pdf')
plt.show()
# %% 绘制角度分布直方图并进行正态检验
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as ss

def ag_cal(x, y):

    ag_list = []

    for i in range(len(x)-1):

        ag = np.rad2deg(np.arctan((y[i+1]-y[i])/(x[i+1]-x[i])))
        ag_list.append(ag)
        i = i+1

    ag_list = np.array(ag_list)

    return ag_list

input_data = ['11', 3, 0.05, 1000]

df = pd.read_excel('profile_data.xlsx', sheet_name=input_data[0], header=0, usecols='A:F', nrows=1001)
x = np.array(df.values[0:input_data[3],0])
y = np.array(df.values[0:input_data[3],input_data[1]])

sampling_interval = input_data[2]
interval_space = sampling_interval/0.05

x1 = x[0::int(interval_space)]
y1 = y[0::int(interval_space)]

angle_list = ag_cal(x1, y1) 

angle_list_mean = np.mean(angle_list)
angle_list_var = np.var(angle_list)
angle_list_std = np.std(angle_list)
# print(angle_list)
y1_norm = np.exp(-(x1 - angle_list_mean) ** 2 / (2 * angle_list_std ** 2)) / (np.sqrt(2*np.pi)*angle_list_std)

# test
ks_test = ss.kstest(angle_list,'norm',args=(angle_list_mean,angle_list_std))
ad_test = ss.anderson(angle_list, dist='norm')
sw_test = ss.shapiro(angle_list)

# print(f"mean angle:{angle_list_mean:.3f}", f"angle var:{angle_list_std:.3f}", ks_test,ad_test,sw_test)
# print(angle_list)

# style
plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '20'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# distribution
fig1 = plt.figure(figsize=(12,8), edgecolor='white', dpi=200)
ax1 = plt.subplot(111)
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['left'].set_linewidth(2)
ax1.set_xlim(-90,90)
ax1.set_ylim(0,0.1)
ax1.set_xticks(np.arange(-90,91,10))
ax1.set_yticks(np.arange(0,0.11,0.01))
ax1.set_ylabel('Probability Density')
ax1.set_xlabel('Angle of inclination/°')
ax1.grid(True,ls='--')
ax1.hist(angle_list, range=(-90,90), bins=180, density=True, alpha=1, color='red', rwidth=0.5)
ax1.plot(np.linspace(-90,90,180),np.exp(-(np.linspace(-90,90,180) - angle_list_mean) ** 2 / (2 * angle_list_std ** 2)) / (np.sqrt(2*np.pi)*angle_list_std),color='green', linestyle='solid', lw=4,label='forward fit')
ax1.text(40,0.045,f'Quartzite No.{input_data[0]} \nProfile No.{input_data[1]} \nSI:{input_data[2]} \nMean={angle_list_mean:.3f} \nStd={angle_list_std:.3f} \nP_value={ks_test[1]:.3f}')

pfplot = fig1.add_axes([.2, .78, .65, .08],facecolor='lightyellow')
pfplot.tick_params(labelsize=12)
pfplot.set_xlim(-1,51)
pfplot.set_ylim(-3,3)
pfplot.set_xlabel('x/mm', fontsize=12)
pfplot.set_ylabel('y/mm', fontsize=12)
pfplot.scatter(x1,y1,marker='^',color='blue',s=2,label='profile')
pfplot.plot(x1,y1,color='blue',lw=1,label='profile')

sorted_ = np.sort(angle_list)
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = ss.norm.ppf(yvals)

qqplot = fig1.add_axes([.2, .4, .2, .3], facecolor='lightcyan')
qqplot.tick_params(labelsize=12)
qqplot.set_xlabel('Theoretical Quantiles', fontsize=12)
qqplot.set_ylabel('Sample Quantiles', fontsize=12)
qqplot.set_title('q-q plot', fontsize=12)
qqplot.scatter(x_label, sorted_, marker='^', color='blueviolet', s=4,label='profile')

plt.savefig('./fig/quartzite_stone/dist_'+str(input_data[0])+'_'+str(input_data[1])+'_'+str(input_data[2])+'.pdf')

# qqplot = fig1.add_axes([.2, .5, .2, .2])
# qqplot.tick_params(labelsize=10)
# ss.probplot(angle_list, dist='norm', plot=qqplot)

# qqplot
# fig2 = plt.figure(figsize=(12,8), edgecolor='white', dpi=200)
# ax2 = plt.subplot(111)
# qqplot = ss.probplot(angle_list, dist='norm', plot=ax2)
# plt.savefig('qq_'+str(input_data[0])+'_'+str(input_data[1])+'_'+str(input_data[2])+'.svg')

plt.show()