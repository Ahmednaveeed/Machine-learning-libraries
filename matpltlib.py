import matplotlib.pyplot as plt
import pandas as pd


# Basic line graph with upto our x and y axis
'''
x = [1,2,3]
y = [2,4,6]
plt.plot(x, y)
plt.title("Basic Graph")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.plot(
    x,y,label='2x',
    color='blue',
    linewidth=2,linestyle='--',
    marker='.',markersize=10,markerfacecolor='red')
plt.plot()
x2 = np.arange(0,4,0.5)
plt.plot(x2, x2**2, 'r', label='x^2')
plt.legend()
plt.savefig('basic_graph.png', dpi=300)
plt.show()
'''

# Basic line graph with fixed x and y axis 
'''
plt.xticks([0,1,2,3,4,5])
plt.yticks([0,2,4,6,8,10])
plt.title(
    "Basic Graph",
    fontdict=
    {'fontsize':20, 'fontname': 'Comic Sans MS'}
    )
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
'''

# Basic bar graph
'''
labels = ['A','B','C','D']
values = [3,7,2,5]
plt.title("Basic Bar Graph")
#plt.bar(labels,values)

bars = plt.bar(labels,values)
bars[0].set_hatch('/')
bars[1].set_color('blue')
bars[2].set_color('green')
bars[3].set_hatch('*')
plt.show()
'''

# Histogram
fifa = pd.read_csv('fifa_data.csv')
plt.hist(fifa.Overall)
plt.show()

###### REAL WORLD EXAMPLE ######
'''
gas = pd.read_csv('gas_prices.csv')
#print(gas.head(5))
plt.figure(figsize=(10, 5))
plt.title("Gas Prices Over Time")
plt.xlabel("Year")
plt.ylabel("Price in USD")
plt.plot(gas.Year,gas.USA,'bo-', label='USA')
plt.plot(gas.Year,gas.Canada,'yo-',label='Canada')
#to print all countries in the dataset
"""
for country in gas:
    if country != 'Year':
        plt.plot(gas.Year, gas[country], marker='.')
"""
plt.xticks(gas.Year[::3])
plt.legend()
plt.show()
'''

