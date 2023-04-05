


def mse_loss(y_t, y_pred):
    loss = 0.0
    for yt, yp in zip(y_t, y_pred):
        loss += (yt - yp) ** 2
    return loss / len(y_t)


class SGD_Optimizer:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


def optimize(optimizer, loss_fn, epochs, batch_size, model, X, y, print_every=100):
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            y_pred = model(X_batch)
            loss = loss_fn(y_batch, y_pred)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % print_every == 0:
            print(f'Epoch: {epoch}, Loss: {loss.data}')

    