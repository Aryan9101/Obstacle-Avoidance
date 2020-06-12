from environment import *
from neural_net import *
import torch


def test(model, num):
    environment = Environment()
    state, _, is_crashed = environment.act(-1)

    t = 0
    environment.reset()

    if num:
        while t < 50:
            environment.reset()
            state, _, is_crashed = environment.act(-1)
            while not is_crashed:
                best_action = torch.argmax(model(state))
                state, reward, is_crashed = environment.act(best_action)
            t += 1
    else:
        while t < 800:
            # time.sleep(0.1)
            best_action = torch.argmax(model(state))
            state, reward, is_crashed = environment.act(best_action)
            t += 1


if __name__ == "__main__":
    model = neural_net()
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    test(model, 0)
