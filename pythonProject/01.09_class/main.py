import matplotlib.pyplot as plt
import pandas as pd

def read_csv(filename):
    data = pd.read_csv(filename)
    return data

def draw(data):
    genders = list(data.gender)
    ages = list(data['age'])
    colors = []
    for i in range(len(genders)):
        if genders[i] == 'm':
            colors.append('blue')
        else:
            colors.append('red')
    plt.bar(range(len(genders)), ages, color=colors)
    plt.show()

if __name__ == '__main__':
    file = "text.csv"
    draw(read_csv(file))