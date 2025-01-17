# TRAINS - Example of manual model configuration and uploading
#
import os
from tempfile import gettempdir

import torch
from trains import Task


task = Task.init(project_name='examples', task_name='Model configuration and upload')

# create a model
model = torch.nn.Module

# Connect a local configuration file
config_file = os.path.join('examples', 'trains', 'reporting', 'data_samples', 'sample.json')
config_file = task.connect_configuration(config_file)
# then read configuration as usual, the backend will contain a copy of it.
# later when executing remotely, the returned `config_file` will be a temporary file
# containing a new copy of the configuration retrieved form the backend
# # model_config_dict = json.load(open(config_file, 'rt'))

# Or Store dictionary of definition for a specific network design
model_config_dict = {
    'value': 13.37,
    'dict': {'sub_value': 'string', 'sub_integer': 11},
    'list_of_ints': [1, 2, 3, 4],
}
model_config_dict = task.connect_configuration(model_config_dict)

# We now update the dictionary after connecting it, and the changes will be tracked as well.
model_config_dict['new value'] = 10
model_config_dict['value'] *= model_config_dict['new value']

# store the label enumeration of the training model
labels = {'background': 0, 'cat': 1, 'dog': 2}
task.connect_label_enumeration(labels)

# storing the model, it will have the task network configuration and label enumeration
print('Any model stored from this point onwards, will contain both model_config and label_enumeration')

torch.save(model, os.path.join(gettempdir(), "model.pt"))
print('Model saved')
