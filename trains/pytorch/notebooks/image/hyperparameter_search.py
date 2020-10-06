#!/usr/bin/env python
# coding: utf-8

# In[3]:


# execute this in command line on all machines to be used as workers before initiating the hyperparamer search 
# ! pip install -U trains-agent==0.15.0
# ! trains-agent daemon --queue default

# pip install with locked versions
# ! pip install -U pandas>=1.0.3
# ! pip install -U trains>=0.16.2
# ! pip install -U optuna==2.0.0


# In[4]:


from trains.automation import UniformParameterRange, UniformIntegerParameterRange
from trains.automation import HyperParameterOptimizer
from trains.automation.optuna import OptimizerOptuna

from trains import Task


# In[5]:


task = Task.init(project_name='Hyperparameter Optimization with Optuna',
                 task_name='Hyperparameter Search',
                 task_type=Task.TaskTypes.optimizer)


# In[7]:


#####################################################################
### Don't forget to replace this default id with your own task id ###
#####################################################################
TEMPLATE_TASK_ID = 'adfb92750a4545b5a73ef857e123f38a'


# In[10]:


optimizer = HyperParameterOptimizer(
    base_task_id=TEMPLATE_TASK_ID,  # This is the experiment we want to optimize
    # here we define the hyper-parameters to optimize
    hyper_parameters=[
        UniformIntegerParameterRange('number_of_epochs', min_value=2, max_value=12, step_size=2),
        UniformIntegerParameterRange('batch_size', min_value=2, max_value=16, step_size=2),
        UniformParameterRange('dropout', min_value=0, max_value=0.5, step_size=0.05),
        UniformParameterRange('base_lr', min_value=0.00025, max_value=0.01, step_size=0.00025),
    ],
    # setting the objective metric we want to maximize/minimize
    objective_metric_title='accuracy',
    objective_metric_series='total',
    objective_metric_sign='max',  # maximize or minimize the objective metric

    # setting optimizer - trains supports GridSearch, RandomSearch, OptimizerBOHB and OptimizerOptuna
    optimizer_class=OptimizerOptuna,
    
    # Configuring optimization parameters
    execution_queue='default',  # queue to schedule the experiments for execution
    max_number_of_concurrent_tasks=2,  # number of concurrent experiments
    optimization_time_limit=60.,  # set the time limit for the optimization process
    compute_time_limit=120,  # set the compute time limit (sum of execution time on all machines)
    total_max_jobs=20,  # set the maximum number of experiments for the optimization. 
                        # Converted to total number of iteration for OptimizerBOHB
    min_iteration_per_job=15000,  # minimum number of iterations per experiment, till early stopping
    max_iteration_per_job=150000,  # maximum number of iterations per experiment
)


# In[ ]:


optimizer.set_report_period(1)  # setting the time gap between two consecutive reports
optimizer.start()  
optimizer.wait()  # wait until process is done
optimizer.stop()  # make sure background optimization stopped


# In[ ]:


# optimization is completed, print the top performing experiments id
k = 3
top_exp = optimizer.get_top_experiments(top_k=k)
print('Top {} experiments are:'.format(k))
for n, t in enumerate(top_exp, 1):
    print('Rank {}: task id={} |result={}'
          .format(n, t.id, t.get_last_scalar_metrics()['accuracy']['total']['last']))


# In[ ]:




