def split(server, client, epoch):
    for e in range(epoch):
        out, labels = client.local_train()
        grad = server.train(out, labels)
        client.backward(grad)
    client_handlers.append(handler)
