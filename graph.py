import matplotlib.pyplot as plt
from models import load_model

def graph(meta: dict, title:str="", xlabel:str="", ylabel:str="") -> None:
    """ Standard graphs, given the meta """
    for key, package in meta.items():
        x_list = []
        y_list = []
        for x, y in package:
            x_list.append(x)
            y_list.append(y)
        plt.plot(x_list, y_list, label=key)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.show()

def graph_model(name:str) -> None:
    """ Graphs the model based on name. """
    payload = load_model(name)
    meta = payload.get("meta", None)
    if meta is None:
        raise Exception("No meta found")
    acc_meta = {
        "Training accuracy": meta.get("training_acc", None),
        "Validation accuracy": meta.get("testing_acc", None)
    }
    graph(acc_meta, title=f"Epochs vs Accuracy of {name}", xlabel="Epochs", ylabel="Accuracy (%)")

    loss = {"loss": meta.get("loss", None)}
    graph(loss, title=f"Epochs vs Loss of {name}", xlabel="Epochs", ylabel="Loss")

if __name__ == '__main__':
    graph_model("linear-classifier")
    graph_model("deep-belief-network")
    graph_model("naive-bayes-model")
    graph_model("restricted-boltzmann-model")
    graph_model("rbm-linear-classifier")