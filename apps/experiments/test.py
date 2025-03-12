datasets = [mnist, cifar10]

clients = distribute(datasets, poison)

global_model = None
clients_model = []

# warmup, boosting

# start
for r in rounds:
    selected_clients = select(clients)
    # trust: calculate trust of each client
    for client in selected_clients:
        client.model = global_model
        client.model.train(client.dataset)
        clients_model[client] = client.model
    # poison detection technique
    global_model = avg(clients_model)
    acc, loss = infer(global_model)
