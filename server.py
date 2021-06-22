# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import flwr as fl
import torch
import mnist
from typing import Tuple
import torch.nn.functional as F

from torchsummary import summary

DATA_ROOT = "./data/pml-training.csv"

def test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, float, float]:
    """Test routine 'Basic MNIST Example'

    Parameters
    ----------
    model: torch.nn.Module :
        Neural network model used in this example.

    test_loader: torch.utils.data.DataLoader :
        DataLoader used in test.

    device: torch.device :
         (Default value = torch.device("cpu"))
         Device where the network will be tested within a client.

    Returns
    -------
        Tuple containing the total number of test samples, the test_loss, and the accuracy evaluated on the test set.

    """
    model.eval()
    
    #print(summary(model, (1, 40)))
    test_loss: float = 0
    correct: int = 0
    num_test_samples: int = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            num_test_samples += len(data)
            #print(data.unsqueeze(1).numpy().shape)
            #pepe
            output = model(data.unsqueeze(1).permute(0, 2, 1))
            test_loss += torch.nn.CrossEntropyLoss()(
                output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= num_test_samples

    return (test_loss, correct / num_test_samples, num_test_samples)

def eval(w):
    train_loader, test_loader = mnist.load_data(
        data_root=DATA_ROOT,
        train_batch_size=64,
        test_batch_size=4,
        cid=5,
        nb_clients=6,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server = mnist.PytorchMNISTClient(
        cid=999,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=1,
        device=device
    )
    server.set_weights(w)
    return test(server.model, train_loader, device)



if __name__ == "__main__":

    strategy = fl.server.strategy.FedAvg(
        eval_fn = eval,
        )

    fl.server.start_server(config={"num_rounds": 10}, strategy = strategy)
