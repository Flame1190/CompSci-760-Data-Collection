# Stereo Improved Neural Supersampling

This is the official implementation of our model: Stereo Improved Neural Supersampling (SINSS). 

# Training
Training is handled by `train.py`. 
Configs can be found under the configs directory. 
Training is commenced by executing

```
python train.py -c configs/<config>.json
```

## Resuming

For certain models we switch the training data format after a certain number of epochs, 
training can be resumed for a model with

```
python train.py -r <model>.pth
```

Note that a config can still be specified with `-c`. If so, the specified config will
partially overwrite (root level keys) the existing config collocated in the same directory as the specified 
model. 

For example, after training `MNSS` on the previous high-resolution frame, training can be switched to use the previous
prediction with

```
python train.py -r <mnss_prev_truth>.pth -c configs/mnss_prev_pred.json
```

**Note:**
NSRR is not a frame recurrent model and as such the `reverse` argument for the dataloader should be set to `false`.

# Testing

Testing is handled by `test.py`. 
There are specific test configs in the config directory that contain a `run` argument. 
This argument controls which testing routine is dispatched.
As with resuming training, the test config will partially overwrite the config the model was trained on.


Example:
```
python test.py -c configs/test_<model>_<scale_factor>.py -r <model>.pth
```

# Data

The stereo dataloaders expect a file structure of `<clip_dir>/<view_dir>/<color|depth|motion>/frame.<png|exr>`.
The single-view dataloaderes expect a file structure of `<clip_dir>/<color|depth|motion>/frame.<png|exr>`.

PNG is the format used for the color data, with EXR being used for the depth and motion data. 

The data modalities are those expected from the unity AOV recorder. 

# Code

All the models, losses, and metrics are found under the `model` directory.

Dataloaders are contained within `data_loaders.py`.

`trainer.py` contains training subroutines for each of the models.