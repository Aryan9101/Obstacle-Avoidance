import torch.optim as optim
from environment import *
from experience_replay import *
from neural_net import *

import pandas as pd

GAMMA = 0.9
BATCH_SIZE = 500
MAX_MEMORY = 50000
EPOCHS = 5000000
STARTING_EPOCH = 0
STARTING_EPSILON = 1


def training(model):
    environment = Environment()

    exp_replay = ExperienceReplay(3, MAX_MEMORY, GAMMA)
    print("Loading memory")
    exp_replay.load_memory(load_memory_from_csv("exp_replay_memory_test.csv"))
    print("Finished loading memory: ", len(exp_replay.get_memory()))

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-9)

    loss_over_time = list()
    running_loss = 0.0

    environment.reset()
    state, _, is_crashed = environment.act(-1)
    states_before_crash = 0

    t = STARTING_EPOCH
    epsilon = max(STARTING_EPSILON - t / EPOCHS, 0.1)
    while True:
        t += 1

        if random.random() <= epsilon:
            action = random.randint(0, 2)
        else:
            q = model(state.to(device))
            action = torch.argmax(q).data

        new_state, reward, is_crashed = environment.act(action)
        exp_replay.remember([state, action, reward, new_state], is_crashed)
        inputs, targets = exp_replay.get_batch_optim(model, device, batch_size=BATCH_SIZE)
        inputs = inputs.to(device)
        targets = targets.to(device)
        state = new_state

        # Model fitting
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        # print(model[0].weight.grad)
        optimizer.step()
        running_loss += loss.item()

        if is_crashed:
            states_before_crash = 0
            environment.reset()
        else:
            states_before_crash += 1

        if t % 1000 == 0:
            loss_over_time.append(running_loss / 1000)
            print("Loss over time: ", loss_over_time)
            running_loss = 0.0

        # epsilon decay
        epsilon = max(epsilon - 1 / EPOCHS, 0.1)

        if t % 10000 == 0:
            torch.save(model.state_dict(), "model_test.pth")
            print("Saving model")
            mem = pd.DataFrame(exp_replay.get_memory())
            mem.to_csv('exp_replay_memory_test.csv', index=False)

        print("Epoch: ", t, "\t| States Before Crash: ", states_before_crash, "\t| Epsilon: ", epsilon,
              "\t| Loss: ", loss.item(), "\t| Distance from Start: ", environment.distance_from_start())


def load_memory_from_csv(filename):
    # Pandas saves a string representation of my SARS' list so I have to reconvert my states back into a tensor
    df = pd.read_csv(filename, delimiter=",")
    df_list = df.values.tolist()
    mem_list = list()

    for line in df_list:
        ns = line[0].strip('][').split(', ')  # ns = new string
        state1 = torch.tensor(
            [float(ns[0][8:]), float(ns[1]), float(ns[2]), float(ns[3]), float(ns[4][:ns[4].index(']')])])
        action = int(ns[5]) if not ns[5][0] == "t" else int(ns[5][7:8])
        reward = float(ns[6])
        state2 = torch.tensor(
            [float(ns[7][8:]), float(ns[8]), float(ns[9]), float(ns[10]), float(ns[11][:ns[11].index(']')])])
        is_crashed = line[1]
        mem_list.append([[state1, action, reward, state2], is_crashed])
    return mem_list


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    model = neural_net().to(device)
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    training(model)
