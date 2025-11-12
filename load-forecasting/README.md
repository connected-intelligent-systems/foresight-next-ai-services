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
curl --request POST --header "Content-Type: application/json" --data @./sample_data/example_request.json --url-query n_samples=100 --url-query return_samples=False http://127.0.0.1:8000/forecast
```

## Expected input format

JSON formatted like [example_request.json](./sample_data/example_request.json)

- POST request with a header: `Content-Type: application/json`
- Time in unix timestamps (UTC, integer, milliseconds)
- Power in Watt
- Example json file in sample_data
- 8 days of history

A simple way is to create `pandas.DataFrame` with unnamed datetime index with a frequency of `1h` and one column by the
name of `power`,
then calling `.to_json()` as is done in [model_provider.py](./src/model_provider.py)
under `if __name__ == '__main__'`

Additionally, the request can have two query parameters:
- `n_samples`: `int`, defaults to `100`, determines the number of Monte-Carlo samples to compute the quantiles from
- `return_samples`: `bool`, defaults to `True`, tells whether to return all samples or the quantiles only


## Output format

JSON file formatted like [example_response.json](./sample_data/example_response.json)

- `samples`: `n` samples for every horizon from `1h` to `24h` sampled from the learned distribution, where `n` is determined by the `n_samples` query parameter.
    This part is omitted if `return_samples` was set to `False`
- `quantiles`: The quantiles `(0.1, 0.25, 0.5, 0.75, 0.9)`computed from the above samples
