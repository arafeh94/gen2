# noinspection PyUnresolvedReferences,PyRedundantParentheses,PyUnboundLocalVariable,PyNoneFunctionAssignment
def cont_fl(data):
    clients_data_per_rounds = distribute(data)  # {'round_1':{'client_1': data, 'client_2': data}}
    for i in rounds:
        # data changes from round to round considering continuous learning
        clients = collect_data(clients_data_per_rounds)
        if initialize_every_x_rounds:
            weights = one_round(clients)
        else:
            cd = measure_clients_drift(clients, keys=(weights[-2], weights[1]))
            if cd > margin:
                weights = one_round(clients)
        iid_clustered_clients = cluster(clients, key=weights)
        main_server = main_server()
        for cluster in iid_clustered_clients:  # in sequence
            server = main_server.get_aggregation_server(cluster)
            for client in iid_clustered_clients:  # in parallel
                split_train(client, server)
            server.aggregate()
            # during update, handle catastrophic forgetting on the client size
            iid_clustered_clients.update(server.aggregated_client_weights)
            # during update, handle catastrophic forgetting on the server size
            main_server.update_weights(server.server_weights)
            # round end


# noinspection PyUnresolvedReferences,PyRedundantParentheses,PyUnboundLocalVariable,PyNoneFunctionAssignment
def distribute(data):
    data_round = dirichlet(data)
    client_data_per_rounds = dirichlet(data_round)
    # example results (targeting label distribution, size distribution, incremental learning)
    client_data_per_rounds = {
        'round_1': {
            'client_1': ['data with dominant label 1'],
            'client_2': ['data with dominant label 2']
        },
        'round_2': {
            'client_1': ['data with dominant label 3'],
            'client_2': ['data with dominant label 1']
        }
    }
    return client_data_per_rounds
