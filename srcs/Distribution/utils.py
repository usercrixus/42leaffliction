import matplotlib.pyplot as plt
import matplotlib.cm as cm


def print_distribution(data):
    # Print repartition
    labels = list(data.keys())
    values = list(data.values())
    plt.figure(figsize=(15, 8))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Fruits Distribution")
    plt.show()

    # Print quantity
    n = len(values)
    colors = cm.get_cmap('tab20', n).colors
    plt.figure(figsize=(15, 8))
    plt.bar(labels, values, color=colors)
    plt.title("Fruits quantity")
    plt.xlabel("Fruits")
    plt.ylabel("Number")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
