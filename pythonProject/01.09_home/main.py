import matplotlib.pyplot as plt
import pandas as pd


def read_csv(filename):
    list_of_data = pd.read_csv(filename)
    return list_of_data


def draw_1(data_table):
    genders = 'female', 'male'
    colors = []
    data_1_1 = data_table.copy()

    for i in range(len(data_table)):
        data_1_1.values[i][0] = 1 - data_1_1.values[i][0]
        data_1_1.values[i][1] = 1 - data_1_1.values[i][1]
    for i in range(len(genders)):
        if genders[i] == 'male':
            colors.append('blue')
        else:
            colors.append('red')

    for i in range(len(data_table)):
        plt.bar(genders, data_table.values[i], color=colors)
        plt.bar(genders, data_1_1.values[i], color='black', bottom=data_table.values[i])
        plt.title("Класс номер " + str(i + 1))
        plt.show()


def draw_2(data_table_2):
    class_num = list(set(data_table_2['Pclass']))
    column = [0, 0, 0]
    data_table_2 = list(data_table_2.Pclass)
    for i in range(len(data_table_2)):
        if data_table_2[i] == 1:
            column[0] += 1
        if data_table_2[i] == 2:
            column[1] += 1
        if data_table_2[i] == 3:
            column[2] += 1
    plt.subplots()
    plt.pie(column, labels=class_num, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()


def draw_3(data_table_):
    genders = list(data_table_.Sex)
    ages = list(data_table_['Age'])
    colors = []
    for i in range(len(data_table_)):
        if genders[i] == 'male':
            colors.append('blue')
        else:
            colors.append('red')
    plt.bar(range(len(genders)), ages, color=colors)
    plt.title("Возростная характеристика")
    plt.show()


if __name__ == '__main__':
    file = "train.csv"
    database = read_csv(file)
    database_1 = database.copy()
    database_2 = database.copy()
    data_base_3 = database.copy()
    database_1 = database_1.pivot_table(values="Survived", index="Pclass", columns="Sex")
    draw_1(database_1)
    draw_2(database_2)
    draw_3(data_base_3)
