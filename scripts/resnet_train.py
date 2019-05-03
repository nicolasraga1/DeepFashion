
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time
import os
import copy

# print("PyTorch Version: ",torch.__version__)
# print("Torchvision Version: ",torchvision.__version__)

print("No Dependency Issues! (So Far)")

# Auxilliary Functions

# CHOOSE THE APPROPRIATE MODEL

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):

    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size



def set_parameter_requires_grad(model, feature_extracting):

    # This should only freeze the fun convolutional bits.
    # FIX THIS FOR THE CONVOLUTIONAL BITS ONLY

    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    return



def set_parameter_conv_only(model, feature_extracting):

    # TODO: SET CONVOLUTIONAL LAYERS ONLY

    return



def train_model(model, dataloaders, criterion, optimizer, num_epochs, model_name, is_inception=False):

    since = time.time()

    all_losses_t = []
    all_acc_t = []

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients - past gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):


                    # Inception has special losses to deal with vanishing gradients
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2

                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                all_losses_t += [loss.item() * inputs.size(0)]
                all_acc_t += [torch.sum(preds == labels.data)/(torch.sum(preds == preds))]

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # dir_name = "./saved_models/" + model_name + "/."
    # model.save(model.save_state(), dir_name)

    return model, val_acc_history, all_losses_t, all_acc_t


# PREPARE THE DATA

## 224 SIZE FOR ALL IMAGES BUT INCEPTION IS 299 = DATALOADER TAKE CARE OF THIS



if __name__ == '__main__':

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    print(" ")

    # SET ALL THE VARIABLES

    data_dir = "./data/"

    #Architecture Considerations

    these_models = ["resnet"]

    output_classes = 50
    num_classes = 50
    feature_extract = True

    # Hyperparameters

    batch_size = 32
    num_epochs = 10

    for model_name in these_models:

        print("Current Model Chosen: {}".format(model_name))

        # Initialize the model for this run
        model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

        # Send the model to GPU
        model_ft = model_ft.to(device)

        data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        }

        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                           shuffle=True, num_workers=4) for x in ['train', 'val']}


        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        print(" ")

        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(params_to_update, lr=0.001)

        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model_ft, hist, losses_t, acc_t = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                 num_epochs, model_name, is_inception=(model_name=="inception"))


        np.save("./resnet_loss", np.array(losses_t))
        np.save("./resnet_acc", np.array(acc_t))
        # print()

        # plt.plot(np.arange(num_epochs), hist)
        # plt.title("Validation Accuracies for ")
