# Using the load forecasting service

You can run the service via the provided image.
The container is exposing port `8000`.

```
docker run -p 8000:8000 ghcr.io/connected-intelligent-systems/foresight-next-ai-services/load-forecasting:latest
```

In this example, it is forwarded to the same port of the local host.
The service is now running and can be interacted with
via synchronous `POST` requests to port `8000` of the machine hosting the container
or any other port the container is forwarding to.
The response contains the model forecast.

You can check whether everything works as intended by sending a test request.
To this end, clone the code then `cd` to the service directory
(the directory containing this README) and execute the following command:

```
curl --request POST --header "Content-Type: application/json" --data @./sample_data/example_request.json http://127.0.0.1:8000/forecast
```

## Expected input format

JSON formatted like [example_request.json](./sample_data/example_request.json)

- POST request with a header: `Content-Type: application/json`
- Time in unix timestamps (UTC, integer, milliseconds)
- Power in Watt
- Example json file in sample_data
- 168 time steps (i.e., 1 week) of history

The previous version had two additional query parameters: `n_samples` and `return_samples`. 
These don't need to be passed and are ignored, now.


## Output format

JSON file formatted like [example_response.json](./sample_data/example_response.json).

## Changes to the previous versions:
- The `samples` key is not returned anymore, only `quantiles`.
- There is now a larger number of quantiles: `(0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)`
- The quantiles are now directly output by the model rather than being computed from samples. 
  As a result, the predictions are significantly faster and more robust.
