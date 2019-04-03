# Gan simple template
## Structure
- Logic
1. Logic Modules are only used together in root python file like train.py. They will not import each other.
2. ``Logger`` only used in ``trainer.py`` to record the training process. In other files, we use ``print``
- Package
1. ``loader``: dataset and dataloader
2. ``models``: networks, loss and metrics
3. ``trainer``: training process, save records(checkpoint, results...)   
    - ``train``
        1. receive each epoch records and log it together.
        2. save checkpoint and visualized results.
    - ``train_epoch``
        1. save records to ``log`` and return.
         
4. ``utils``: frequently used functions, arguments
- Execution
1. ``train.py``: main train logic
2. ``debug.py``: test each package independently
## Reference
https://github.com/victoresque/pytorch-template