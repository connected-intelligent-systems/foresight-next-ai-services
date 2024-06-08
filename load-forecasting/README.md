# Using the training and prediction service

The service can be accessed in one of two ways:
1. By instantiating a container from the image provided via the GitLab image registry:
```
registry.gitlab.com/team-ft/load-forecasting1/darts-model-serving:cpu-latest
```

2. By cloning the repo and running the script
```
python3 -m src.flask_server <arguments>
```
inside a python 10 environment with all packages from [requirements.txt](requirements.txt) installed.

The arguments passed to the service are the same, regardless of how it is launched.
The example provided below explains the workflow for the container.
When running the script directly both the port forwarding and the volume mount should be omitted.
Instead `POST` request can be addressed directly to port `5001` of `localhost`
and local paths can be passed as launch arguments in the command line.


## Accessing the image

This example uses podman but the same commands should work identically in docker.
1. Login to access the image registry using the username and password from the token:

```
podman login registry.gitlab.com
```

It will prompt for a user name and a password. Use the ones from the token provided via email.

2. Pull the image:

```
podman pull registry.gitlab.com/team-ft/load-forecasting1/darts-model-serving:cpu-latest
```


## Running in prediction mode

To run the image locally forwarding the port `5001`:

```
podman run -p 5001:5001 registry.gitlab.com/team-ft/load-forecasting1/darts-model-serving:cpu-latest
```

To check whether everything works as intended you can send a test request to the service.
Clone the code from GitLab, then `cd` to project root and execute the following command:

```
curl http://127.0.0.1:5001/forecast --request POST --header "Content-Type: application/json" --data @./sample_data/example_request.json
```

By default this will use the global pre-trained model that is baked into the image.
We can, however, use any model stored with `darts.models.forecasting.torch_forecasting_model.TorchForecastingModel.save()`,
in particular, ones trained in this container.
In order to instruct the container to use an alternative model, it needs to be placed in a volume that is then mounted by the container.
For a minimal example, we can create a local directory and copy a local model there.
```
mkdir ./tmp 
cp -r ./models/spread_val_set_long ./tmp/alt_model
```

Now we start another container, but this time, we mount the newly created directory as a volume
and pass the argument `-s` or `--source-model` with a path pointing to the parent directory of the model.
Note, that this directory should not contain anything but the saved model (one file without extension and one .ckpt file)

```
podman run -p 5001:5001 -v ./tmp:/mnt registry.gitlab.com/team-ft/load-forecasting1/darts-model-serving:cpu-latest --source-model /mnt/alt_model
```

The output log should now have a line "`Loading model spread_val_set_long...`".
Note, that the model name has changed from `global` too `spread_val_set_long`,
which is the name of the actual save we copied, rather than it's parent directory.


Also, make sure that all the optargs to `podman run` come before the actual image name,
otherwise they will be passed to the script instead and cause errors.

The service should now be up and running and can be interacted with
via synchronous `POST` requests to port `5001` of the machine hosting the container
or any other port the container is forwarding to.


### Expected input format

JSON formatted like [example_request.json](sample_data%2Fexample_request.json)

- POST request with a header: `Content-Type: application/json`
- Time in unix timestamps (UTC, integer, milliseconds)
- Power in Watt
- Example json file in sample_data
- 8 days of history

A simple way is to create `pandas.Dataframe` with unnamed datetime index with a frequency of `1h` and one column by the
name of `power`,
then calling `.to_json()` as is done in [model_provider.py](src%2Fmodel_provider.py)
under `if __name__ == '__main__'`


### Output format

JSON file formatted like [example_response.json](sample_data%2Fexample_response.json)

- samples: 100 samples for every horizon from `1h` to `24h` sampled from the learned distribution.
- quantiles: The quantiles `(0.1, 0.25, 0.5, 0.75, 0.9)`computed from the above samples


## Running in training mode

To run the service in training mode one can use the same image or code as for prediction.
To trigger a model training, one has to pass two additional arguments:
- `-d` or `--dest-model` should provide a path pointing where the new model version should be saved.
Since container storage is ephemeral, this path should point somewhere on the mounted volume.
- `-t` or `--training-data` should provide a path to a data set on which the model should be trained.
More details are given in the next subsection.
- `-s` or `--source-model` can be provided optionally.
In this mode, it should point to the parent directory of a save whose weights should be used as starting point for the training.

Note, that if one of the arguments `-d` or `-t` is provided, the other one is mandatory.
If this criterion is fulfilled, the container will create the parent directory for the new model save
checking for permissions and path sanity.
If no errors are encountered, a training process will be launched and a model saved upon completion.
After that, the container will terminate.
In this mode, no flask server will be started.

### Expected format for training data
The training data should be a single .csv file formatted like
[ccef3bd47b7d8a93578791579362d7f61e6daed3dd4acbab47997bc8927218f.csv](sample_data%ccef3bd47b7d8a93578791579362d7f61e6daed3dd4acbab47997bc8927218f.csv).


In particular, this means:
- The header `timestamp,power`
- A datetime index
- An hourly sampling rate
- Missing values being denoted by empty strings as can be seen in lines 14258-14421.

If the data set contains missing values it is treated as several time series.
If this is not the desired behavior (especially for smaller gaps)
an imputation has to be performed beforehand.
