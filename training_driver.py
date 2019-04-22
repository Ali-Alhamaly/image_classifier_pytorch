import torch
#from torch import nn
import time
from copy import deepcopy

def testing(model, testloader, criterion,device):
    model.eval()
    with torch.no_grad():
        
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model=model.to(device)
        test_accuracy = 0
        test_loss = 0
        counter = 0
        for images, labels in testloader:
            counter+=1
            images , labels = images.to(device), labels.to(device)
        
        
        
            output = model(images)
            test_loss += criterion(output, labels).item()
        
            ## Calculating the accuracy
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            test_accuracy += equality.type_as(torch.FloatTensor()).mean()
            print("Batch: {}/{}.. ".format(counter, len(testloader)),
              "Testing Loss: {:.3f}.. ".format(criterion(output, labels).item()),
              "Testing  Accuracy: {:.3f}%".format(100*equality.type_as(torch.FloatTensor()).mean()))
        
        print("Testing Loss aggregate: {:.3f}.. ".format(test_loss/len(testloader)),
              "Testing  Accuracy aggregate: {:.3f}%".format(100*test_accuracy/len(testloader)))
    return test_accuracy/len(testloader)


def validation(model, validloader, criterion,device):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    accuracy = 0
    valid_loss = 0
    for images, labels in validloader:
        images , labels = images.to(device), labels.to(device)



        output = model(images)
        valid_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return valid_loss/len(validloader), accuracy/len(validloader)


def training(model, trainloader, validloader, criterion, optimizer,scheduler,device,epochs=4):
    
    since = time.time()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_acc_history = []
    best_acc = 0.0
    best_model_wts = deepcopy(model.state_dict())


    for e in range(epochs):

        # Model in training mode, dropout is on
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images , labels = images.to(device), labels.to(device)


            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else: # validation phase
            # Model in inference mode, dropout is off
            model.eval()

            # Turn off gradients for validation, will speed up inference
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validloader, criterion,device)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Validation Loss: {:.3f}.. ".format(valid_loss),
                  "Validation Accuracy: {:.3f}%".format(100*accuracy))

            
            # deep copy the model
            if accuracy > best_acc:
                best_acc = accuracy
                best_model_wts = deepcopy(model.state_dict())

            val_acc_history.append(accuracy)
            scheduler.step(valid_loss)

            # Make sure dropout and grads are on for training
            model.train()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

